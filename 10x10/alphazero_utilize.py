import logging  
import math  
import os  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from tqdm import tqdm  
from collections import deque, defaultdict  
from random import shuffle  
import wandb  
import yaml  
from multiprocessing import Pool  
import functools  
from torch.utils.data import DataLoader, TensorDataset  
import threading  
from queue import Queue  
import time  
  
import game  
  
logging.basicConfig(level=logging.INFO)  
log = logging.getLogger(__name__)  
  
  
class MCTS:  
    """  
    Optimized MCTS with batched predictions, memory management, and optional async mode.  
    """  
  
    def __init__(self, game, nnet, args):  
        self.game = game  
        self.nnet = nnet  
        self.args = args  
        # Use defaultdict for memory efficiency  
        self.Qsa = defaultdict(float)  
        self.Nsa = defaultdict(int)  
        self.Ns = defaultdict(int)  
        self.Ps = {}  
        self.Es = {}  
        self.Vs = {}  
          
        # Batch prediction optimization  
        self.prediction_batch = []  
        self.batch_size = getattr(args, 'prediction_batch_size', 32)  
        self.leaf_results = {}  # Store results for batched leaf nodes  
          
        # Memory management  
        self.max_states = getattr(args, 'max_mcts_states', 100000)  
        self.state_counter = 0  
          
        # Async MCTS support  
        self.async_mode = getattr(args, 'async_mcts', False)  
        if self.async_mode:  
            self.virtual_loss = getattr(args, 'virtual_loss', 3)  
            self.lock = threading.Lock()  
            self.prediction_queue = Queue(maxsize=100)  
            self.result_queue = Queue()  
            self.prediction_thread = None  
            self.start_prediction_worker()  
  
    def start_prediction_worker(self):  
        """Start a dedicated thread for neural network predictions in async mode."""  
        def prediction_worker():  
            while True:  
                batch = []  
                while not self.prediction_queue.empty() and len(batch) < self.batch_size:  
                    try:  
                        item = self.prediction_queue.get_nowait()  
                        batch.append(item)  
                    except:  
                        break  
                  
                if batch:  
                    boards = torch.FloatTensor(np.array([item[0] for item in batch]))  
                    if self.args.cuda:  
                        boards = boards.cuda()  
                      
                    self.nnet.eval()  
                    with torch.no_grad():  
                        pis, vs = self.nnet(boards)  
                      
                    for i, (board, s, callback) in enumerate(batch):  
                        self.result_queue.put((s, torch.exp(pis[i]).cpu().numpy(),   
                                              vs[i].cpu().numpy()[0]))  
                else:  
                    time.sleep(0.001)  # Small delay to prevent busy waiting  
          
        self.prediction_thread = threading.Thread(target=prediction_worker, daemon=True)  
        self.prediction_thread.start()  
  
    def getActionProb(self, canonicalBoard, temp=1):  
        """  
        Performs numMCTSSims simulations with optimized batched predictions.  
        """  
        # Clear previous batch data  
        if not self.async_mode:  
            self.prediction_batch = []  
            self.leaf_results = {}  
          
        for _ in range(self.args.numMCTSSims):  
            self.search(canonicalBoard)  
          
        # Process any remaining batch (sync mode only)  
        if not self.async_mode and self.prediction_batch:  
            self._process_prediction_batch()  
  
        s = self.game.stringRepresentation(canonicalBoard)  
        counts = [  
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0  
            for a in range(self.game.getActionSize())  
        ]  
  
        if temp == 0:  
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()  
            bestA = np.random.choice(bestAs)  
            probs = [0] * len(counts)  
            probs[bestA] = 1  
            return probs  
  
        counts = [x ** (1.0 / temp) for x in counts]  
        counts_sum = float(sum(counts))  
        probs = [x / counts_sum for x in counts]  
        return probs  
  
    def search(self, canonicalBoard):  
        """  
        Optimized MCTS search with batched leaf node processing.  
        """  
        s = self.game.stringRepresentation(canonicalBoard)  
          
        # Memory cleanup  
        self.state_counter += 1  
        if self.state_counter % self.max_states == 0:  
            self._cleanup_old_states()  
  
        if s not in self.Es:  
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)  
        if self.Es[s] is not None:  
            return self.Es[s]  
  
        if self.async_mode:  
            return self._search_async(canonicalBoard, s)  
        else:  
            return self._search_sync(canonicalBoard, s)  
      
    def _search_sync(self, canonicalBoard, s):  
        """Synchronous search with batched predictions."""  
        if s not in self.Ps:  
            # Add to batch for prediction  
            self.prediction_batch.append((canonicalBoard.copy(), s))  
              
            # Process batch if full  
            if len(self.prediction_batch) >= self.batch_size:  
                self._process_prediction_batch()  
                return self._continue_search(canonicalBoard, s)  
            else:  
                # Return temporary value, will be updated after batch processing  
                return 0.0  
  
        valids = self.Vs[s]  
        cur_best = -float("inf")  
        best_act = -1  
  
        # UCB selection  
        for a in range(self.game.getActionSize()):  
            if valids[a]:  
                u = self.Qsa.get((s, a), 0) + self.args.cpuct * self.Ps[s][  
                    a  
                ] * math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0))  
  
                if u > cur_best:  
                    cur_best = u  
                    best_act = a  
  
        a = best_act  
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)  
        next_s = self.game.getCanonicalForm(next_s, next_player)  
  
        v = -self.search(next_s)  
  
        # Update statistics  
        if (s, a) in self.Qsa:  
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (  
                self.Nsa[(s, a)] + 1  
            )  
            self.Nsa[(s, a)] += 1  
        else:  
            self.Qsa[(s, a)] = v  
            self.Nsa[(s, a)] = 1  
  
        self.Ns[s] += 1  
        return v  
      
    def _search_async(self, canonicalBoard, s):  
        """Asynchronous search with virtual loss."""  
        with self.lock:  
            if s not in self.Ps:  
                # Apply virtual loss to discourage other threads from exploring this node  
                self.Ns[s] = self.Ns.get(s, 0) + self.virtual_loss  
                  
                # Queue for prediction  
                self.prediction_queue.put((canonicalBoard.copy(), s, None))  
                  
                # Try to get result if available  
                try:  
                    result_s, pi, v = self.result_queue.get_nowait()  
                    if result_s == s:  
                        # Remove virtual loss and set real values  
                        self.Ns[s] -= self.virtual_loss  
                        valids = self.game.getValidMoves(canonicalBoard, 1)  
                        pi = pi * valids  
                        sum_Ps_s = np.sum(pi)  
                        if sum_Ps_s > 0:  
                            pi /= sum_Ps_s  
                        else:  
                            log.error("All valid moves were masked, doing a workaround.")  
                            pi = pi + valids  
                            pi /= np.sum(pi)  
                          
                        self.Ps[s] = pi  
                        self.Vs[s] = valids  
                        self.Ns[s] = 0  
                        return v  
                except:  
                    pass  
                  
                return 0.0  # Return temporary value  
  
        # Rest of the search logic (same as sync)  
        valids = self.Vs[s]  
        cur_best = -float("inf")  
        best_act = -1  
          
        for a in range(self.game.getActionSize()):  
            if valids[a]:  
                u = self.Qsa.get((s, a), 0) + self.args.cpuct * self.Ps[s][  
                    a  
                ] * math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0))  
                  
                if u > cur_best:  
                    cur_best = u  
                    best_act = a  
          
        a = best_act  
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)  
        next_s = self.game.getCanonicalForm(next_s, next_player)  
          
        v = -self.search(next_s)  
          
        with self.lock:  
            if (s, a) in self.Qsa:  
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (  
                    self.Nsa[(s, a)] + 1  
                )  
                self.Nsa[(s, a)] += 1  
            else:  
                self.Qsa[(s, a)] = v  
                self.Nsa[(s, a)] = 1  
              
            self.Ns[s] += 1  
          
        return v  
      
    def _process_prediction_batch(self):  
        """Process a batch of leaf states through the neural network."""  
        if not self.prediction_batch:  
            return  
          
        boards = torch.FloatTensor(np.array([state[0] for state in self.prediction_batch]))  
        if self.args.cuda:  
            boards = boards.cuda()  
          
        self.nnet.eval()  
        with torch.no_grad():  
            pis, vs = self.nnet(boards)  
          
        # Store results  
        for i, (board, s) in enumerate(self.prediction_batch):  
            self.leaf_results[s] = (  
                torch.exp(pis[i]).cpu().numpy(),  
                vs[i].cpu().numpy()[0]  
            )  
          
        self.prediction_batch = []  
      
    def _continue_search(self, canonicalBoard, s):  
        """Continue search after batch prediction."""  
        if s in self.leaf_results:  
            pi, v = self.leaf_results[s]  
            valids = self.game.getValidMoves(canonicalBoard, 1)  
            pi = pi * valids  
            sum_Ps_s = np.sum(pi)  
            if sum_Ps_s > 0:  
                pi /= sum_Ps_s  
            else:  
                log.error("All valid moves were masked, doing a workaround.")  
                pi = pi + valids  
                pi /= np.sum(pi)  
              
            self.Ps[s] = pi  
            self.Vs[s] = valids  
            self.Ns[s] = 0  
            return v  
        return 0.0  
      
    def _cleanup_old_states(self):  
        """Clean up old MCTS states to prevent memory bloat."""  
        # Keep only recently visited states  
        if len(self.Ns) > self.max_states:  
            # Sort by visit count and keep top states  
            sorted_states = sorted(self.Ns.items(), key=lambda x: x[1], reverse=True)  
            keep_states = set([state for state, _ in sorted_states[:self.max_states//2]])  
              
            # Clean up dictionaries  
            for key in list(self.Ps.keys()):  
                if key not in keep_states:  
                    del self.Ps[key]  
            for key in list(self.Es.keys()):  
                if key not in keep_states:  
                    del self.Es[key]  
            for key in list(self.Vs.keys()):  
                if key not in keep_states:  
                    del self.Vs[key]  
  
  
class GomokuNNet(nn.Module):  
    """Unchanged network architecture as requested."""  
    def __init__(self, game, args):  
        self.board_x, self.board_y = game.getBoardSize()  
        self.action_size = game.getActionSize()  
        self.args = args  
  
        super(GomokuNNet, self).__init__()  
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(  
            args.num_channels, args.num_channels, 3, stride=1, padding=1  
        )  
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)  
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)  
  
        self.bn1 = nn.BatchNorm2d(args.num_channels)  
        self.bn2 = nn.BatchNorm2d(args.num_channels)  
        self.bn3 = nn.BatchNorm2d(args.num_channels)  
        self.bn4 = nn.BatchNorm2d(args.num_channels)  
  
        self.fc1 = nn.Linear(  
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024  
        )  
        self.fc_bn1 = nn.BatchNorm1d(1024)  
  
        self.fc2 = nn.Linear(1024, 512)  
        self.fc_bn2 = nn.BatchNorm1d(512)  
  
        self.fc3 = nn.Linear(512, self.action_size)  
        self.fc4 = nn.Linear(512, 1)  
  
    def forward(self, s):  
        s = s.view(-1, 1, self.board_x, self.board_y)  
        s = F.relu(self.bn1(self.conv1(s)))  
        s = F.relu(self.bn2(self.conv2(s)))  
        s = F.relu(self.bn3(self.conv3(s)))  
        s = F.relu(self.bn4(self.conv4(s)))  
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))  
  
        s = F.dropout(  
            F.relu(self.fc_bn1(self.fc1(s))),  
            p=self.args.dropout,  
            training=self.training,  
        )  
        s = F.dropout(  
            F.relu(self.fc_bn2(self.fc2(s))),  
            p=self.args.dropout,  
            training=self.training,  
        )  
  
        pi = self.fc3(s)  
        v = self.fc4(s)  
  
        return F.log_softmax(pi, dim=1), torch.tanh(v)  
  
  
class AverageMeter(object):  
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""  
    def __init__(self):  
        self.val = 0  
        self.avg = 0  
        self.sum = 0  
        self.count = 0  
  
    def __repr__(self):  
        return f"{self.avg:.2e}"  
  
    def update(self, val, n=1):  
        self.val = val  
        self.sum += val * n  
        self.count += n  
        self.avg = self.sum / self.count  
  
  
class NNetWrapper:
    """Optimized neural network wrapper with DataLoader and pin_memory."""

    def __init__(self, game, args):
        self.nnet = GomokuNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def __call__(self, boards):
        boards = torch.tensor(boards, dtype=torch.float32, device=self.device)
        return self.nnet(boards)
        self.args = args

        # Device selection: CUDA, MPS, or CPU
        if hasattr(args, 'cuda') and args.cuda:
            self.nnet.cuda()
            self.device = torch.device('cuda')
        elif hasattr(args, 'mps') and args.mps:
            self.nnet.to('mps')
            self.device = torch.device('mps')
        else:
            self.nnet.to('cpu')
            self.device = torch.device('cpu')

    def eval(self):
        self.nnet.eval()

    # Move optimizer and step initialization to __init__
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.max_lr)
        self.total_steps = self.args.numIters * self.args.epochs * (self.args.maxlenOfQueue // self.args.batch_size)
        self.current_step = 0

    # Call optimizer initialization in __init__
    def __init__(self, game, args):
        self.nnet = GomokuNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Device selection: CUDA, MPS, or CPU
        if hasattr(args, 'cuda') and args.cuda:
            self.nnet.cuda()
            self.device = torch.device('cuda')
        elif hasattr(args, 'mps') and args.mps:
            self.nnet.to('mps')
            self.device = torch.device('mps')
        else:
            self.nnet.to('cpu')
            self.device = torch.device('cpu')

        self._init_optimizer()
  
    def get_learning_rate(self):  
        """Implement 1cycle learning rate strategy"""  
        if self.current_step >= self.total_steps:  
            return self.args.min_lr  
          
        half_cycle = self.total_steps // 2  
          
        if self.current_step <= half_cycle:  
            phase = self.current_step / half_cycle  
            lr = self.args.min_lr + (self.args.max_lr - self.args.min_lr) * phase  
        else:  
            phase = (self.current_step - half_cycle) / half_cycle  
            lr = self.args.max_lr - (self.args.max_lr - self.args.min_lr) * phase  
          
        return lr  
  
    def train(self, examples):  
        """  
        Optimized training with DataLoader and pin_memory for faster GPU transfers.  
        """  
        # Convert to torch dataset  
        boards = torch.FloatTensor(np.array([x[0] for x in examples]))  
        target_pis = torch.FloatTensor(np.array([x[1] for x in examples]))  
        target_vs = torch.FloatTensor(np.array([x[2] for x in examples]))  
          
        dataset = TensorDataset(boards, target_pis, target_vs)  
        dataloader = DataLoader(  
            dataset,   
            batch_size=self.args.batch_size,  
            shuffle=True,  
            num_workers=getattr(self.args, 'num_workers', 4),  
            pin_memory=getattr(self.args, 'pin_memory', True)  
        )  
          
        for epoch in range(self.args.epochs):  
            print("EPOCH ::: " + str(epoch + 1))  
            self.nnet.train()  
            pi_losses = AverageMeter()  
            v_losses = AverageMeter()  
  
            t = tqdm(dataloader, desc="Training Net")  
            for boards, target_pis, target_vs in t:  
                # Update learning rate  
                lr = self.get_learning_rate()  
                for param_group in self.optimizer.param_groups:  
                    param_group['lr'] = lr  
                self.current_step += 1  
  
                # Move tensors to correct device
                if hasattr(self, 'device'):
                    boards, target_pis, target_vs = boards.to(self.device), target_pis.to(self.device), target_vs.to(self.device)
  
                # Compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)  
                l_v = self.loss_v(target_vs, out_v)  
                total_loss = l_pi + l_v  
  
                # Record loss  
                pi_losses.update(l_pi.item(), boards.size(0))  
                v_losses.update(l_v.item(), boards.size(0))  
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, lr=f"{lr:.1e}")  
  
                # Compute gradient and do SGD step  
                self.optimizer.zero_grad()  
                total_loss.backward()  
                  
                if self.args.grad_clip:  
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.args.grad_clip)  
                  
                self.optimizer.step()  
  
                if getattr(self.args, 'wandb', False):  
                    wandb.log({  
                        'learning_rate': lr,  
                        'policy_loss': l_pi.item(),  
                        'value_loss': l_v.item(),  
                        'total_loss': total_loss.item(),  
                        'current_step': self.current_step,  
                    })  
  
    def predict(self, board):  
        """  
        Single board prediction (kept for compatibility).  
        """  
        board = torch.FloatTensor(board.astype(np.float32))
        if hasattr(self, 'device'):
            board = board.to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
  
    def loss_pi(self, targets, outputs):  
        return -torch.sum(targets * outputs) / targets.size()[0]  
  
    def loss_v(self, targets, outputs):  
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]  
  
    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):  
        filepath = os.path.join(folder, filename)  
        if not os.path.exists(folder):  
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))  
            os.mkdir(folder)  
        else:  
            print("Checkpoint Directory exists! ")  
        torch.save({"state_dict": self.nnet.state_dict()}, filepath)  
  
    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):  
        folder = folder.rstrip('/')  
        filepath = os.path.join(folder, filename)  
        if not os.path.exists(filepath):  
            raise ValueError("No model in path {}".format(filepath))  
        map_location = None if self.args.cuda else "cpu"  
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)  
        self.nnet.load_state_dict(checkpoint["state_dict"])  
  
  
def execute_episode_parallel(args):  
    """Worker function for parallel self-play episodes."""  
    game, nnet_args = args
    nnet_args = dotdict(nnet_args)
    nnet = NNetWrapper(game, nnet_args)
    mcts = MCTS(game, nnet, nnet_args)
    self_play = SelfPlay(game, nnet, nnet_args)
    return self_play.executeEpisode()
  
  
class SelfPlay:  
    """  
    Optimized self-play with parallel episode execution.  
    """  
    def __init__(self, game, nnet, args):  
        self.game = game  
        self.nnet = nnet  
        self.pnet = self.nnet.__class__(self.game, args)  
        self.args = args  
        self.mcts = MCTS(self.game, self.nnet, self.args)  
        self.trainExamplesHistory = []  
  
    def executeEpisode(self):  
        """  
        Executes one episode of self-play (unchanged logic).  
        """  
        trainExamples = []  
        board = self.game.getInitBoard()  
        self.curPlayer = 1  
        episodeStep = 0  
  
        while True:  
            episodeStep += 1  
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)  
            temp = int(episodeStep < self.args.tempThreshold)  
  
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)  
            sym = self.game.getSymmetries(canonicalBoard, pi)  
            for b, p in sym:  
                trainExamples.append([b, self.curPlayer, p, None])  
  
            action = np.random.choice(len(pi), p=pi)  
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)  
  
            r = self.game.getGameEnded(board, self.curPlayer)  
  
            if r is not None:  
                return [  
                    (x[0], x[2], r * (1 if self.curPlayer == x[1] else -1))  
                    for x in trainExamples  
                ]  
  
    def learn(self):  
        """  
        Optimized learning with parallel self-play episodes.  
        """  
        num_workers = getattr(self.args, 'num_workers', 4)  
          
        for i in range(1, self.args.numIters + 1):  
            log.info(f"Starting Iter #{i} ...")  
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)  
              
            # Execute episodes in parallel  
            chunk_size = max(1, self.args.numEps // num_workers)  
            # Convert dotdict to dict for multiprocessing compatibility
            chunks = [(self.game, dict(self.args)) for _ in range(chunk_size)]
              
            with Pool(processes=num_workers) as pool:  
                for chunk_results in pool.imap_unordered(execute_episode_parallel, chunks, chunksize=1):  
                    iterationTrainExamples += chunk_results  
  
            # Save the iteration examples to the history  
            self.trainExamplesHistory.append(iterationTrainExamples)  
  
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:  
                log.warning(  
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"  
                )  
                self.trainExamplesHistory.pop(0)  
  
            # Shuffle examples before training  
            trainExamples = []  
            for e in self.trainExamplesHistory:  
                trainExamples.extend(e)  
            shuffle(trainExamples)  
  
            # Training new network, keeping a copy of the old one  
            self.nnet.save_checkpoint(  
                folder=self.args.checkpoint, filename="temp.pth.tar"  
            )  
            self.pnet.load_checkpoint(  
                folder=self.args.checkpoint, filename="temp.pth.tar"  
            )  
            pmcts = MCTS(self.game, self.pnet, self.args)  
  
            self.nnet.train(trainExamples)  
            nmcts = MCTS(self.game, self.nnet, self.args)  
  
            log.info("PITTING AGAINST PREVIOUS VERSION")  
            arena = game.Arena(  
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),  
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),  
                self.game,  
            )  
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)  
  
            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))  
            if (  
                pwins + nwins == 0  
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold  
            ):  
                log.info("REJECTING NEW MODEL")  
                self.nnet.load_checkpoint(  
                    folder=self.args.checkpoint, filename="temp.pth.tar"  
                )  
            else:  
                log.info("ACCEPTING NEW MODEL")  
                self.nnet.save_checkpoint(  
                    folder=self.args.checkpoint, filename="best.pth.tar"  
                )  
  
  
class dotdict(dict):  
    def __getattr__(self, name):
        # Delegate special attributes to base class
        if name.startswith('__') and name.endswith('__'):
            return super().__getattribute__(name)
        return self[name]

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.clear()
        self.update(state)
  
  
def load_config(config_path):
        # Add aliases for expected top-level keys from flattened config
        top_level_aliases = {
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'num_iterations': 'training.num_iterations',
            'num_episodes': 'training.num_episodes',
            'max_queue_length': 'training.max_queue_length',
            'num_iters_history': 'training.num_iters_history',
            'update_threshold': 'training.update_threshold',
            'arena_compare': 'training.arena_compare',
            'temp_threshold': 'training.temp_threshold',
            'num_channels': 'network.num_channels',
            'dropout': 'network.dropout',
            'grad_clip': 'network.grad_clip',
            'board_size': 'game.board_size',
            'checkpoint_dir': 'system.checkpoint_dir',
            'load_model': 'system.load_model',
            'load_folder_file': 'system.load_folder_file',
        }
        for alias, flat in top_level_aliases.items():
            if flat in flat_args:
                flat_args[alias] = flat_args[flat]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Recursively flatten all config sections into a single dict for dotdict
        def flatten_dict(d, parent_key='', sep='.'): 
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        flat_args = flatten_dict(config)

        # Add legacy key aliases for backward compatibility
        legacy_aliases = {
            'numIters': 'num_iterations',
            'numEps': 'num_episodes',
            'maxlenOfQueue': 'max_queue_length',
            'numItersForTrainExamplesHistory': 'num_iters_history',
            'updateThreshold': 'update_threshold',
            'arenaCompare': 'arena_compare',
            'tempThreshold': 'temp_threshold',
        }
        for legacy, new in legacy_aliases.items():
            if new in flat_args:
                flat_args[legacy] = flat_args[new]

        # Add min_lr and max_lr legacy aliases for learning rate
        if 'learning_rate.min' in flat_args:
            flat_args['min_lr'] = flat_args['learning_rate.min']
        if 'learning_rate.max' in flat_args:
            flat_args['max_lr'] = flat_args['learning_rate.max']

        # Device logic
        flat_args['cuda'] = flat_args.get('cuda', False) and torch.cuda.is_available()
        flat_args['mps'] = flat_args.get('mps', True) and torch.backends.mps.is_available()
        flat_args['device'] = 'cuda' if flat_args['cuda'] else ('mps' if flat_args['mps'] else 'cpu')
        flat_args['load_folder_file'] = tuple(flat_args.get('load_folder_file', ['./temp', 'best.pth.tar']))

        # Optimization defaults
        flat_args['prediction_batch_size'] = flat_args.get('prediction_batch_size', 32)
        flat_args['max_mcts_states'] = flat_args.get('max_mcts_states', 100000)
        flat_args['num_workers'] = flat_args.get('num_workers', 4)
        flat_args['pin_memory'] = flat_args.get('pin_memory', True)
        flat_args['async_mcts'] = flat_args.get('async_mcts', False)
        flat_args['virtual_loss'] = flat_args.get('virtual_loss', 3)

        return dotdict(flat_args)
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    flat_args = flatten_dict(config)

    # Add legacy key aliases for backward compatibility
    legacy_aliases = {
        'numIters': 'num_iterations',
        'numEps': 'num_episodes',
        'maxlenOfQueue': 'max_queue_length',
        'numItersForTrainExamplesHistory': 'num_iters_history',
        'updateThreshold': 'update_threshold',
        'arenaCompare': 'arena_compare',
        'tempThreshold': 'temp_threshold',
    }
    for legacy, new in legacy_aliases.items():
        if new in flat_args:
            flat_args[legacy] = flat_args[new]

    # Add min_lr and max_lr legacy aliases for learning rate
    if 'learning_rate.min' in flat_args:
        flat_args['min_lr'] = flat_args['learning_rate.min']
    if 'learning_rate.max' in flat_args:
        flat_args['max_lr'] = flat_args['learning_rate.max']

    # Device logic
    flat_args['cuda'] = flat_args.get('cuda', False) and torch.cuda.is_available()
    flat_args['mps'] = flat_args.get('mps', True) and torch.backends.mps.is_available()
    flat_args['device'] = 'cuda' if flat_args['cuda'] else ('mps' if flat_args['mps'] else 'cpu')
    flat_args['load_folder_file'] = tuple(flat_args.get('load_folder_file', ['./temp', 'best.pth.tar']))

    # Optimization defaults
    flat_args['prediction_batch_size'] = flat_args.get('prediction_batch_size', 32)
    flat_args['max_mcts_states'] = flat_args.get('max_mcts_states', 100000)
    flat_args['num_workers'] = flat_args.get('num_workers', 4)
    flat_args['pin_memory'] = flat_args.get('pin_memory', True)
    flat_args['async_mcts'] = flat_args.get('async_mcts', False)
    flat_args['virtual_loss'] = flat_args.get('virtual_loss', 3)

    return dotdict(flat_args)
  
  
def print_config(args):  
    """Pretty print the configuration"""  
    print("\n=== Configuration ===")  
    print("Training Parameters:")  
    print(f"  Epochs: {args.epochs}")  
    print(f"  Batch Size: {args.batch_size}")  
    print(f"  Number of Iterations: {args.numIters}")  
    print(f"  Episodes per Iteration: {args.numEps}")  
    print(f"  Max Queue Length: {args.maxlenOfQueue}")  
    print(f"  Training History Length: {args.numItersForTrainExamplesHistory}")  
    print(f"  Update Threshold: {args.updateThreshold}")  
    print(f"  Arena Compare Games: {args.arenaCompare}")  
    print(f"  Temperature Threshold: {args.tempThreshold}")  
      
    print("\nNetwork Parameters:")  
    print(f"  Number of Channels: {args.num_channels}")  
    print(f"  Dropout: {args.dropout}")  
    print(f"  Learning Rate Range: {args.min_lr} - {args.max_lr}")  
    print(f"  Gradient Clip: {args.grad_clip}")  
      
    print("\nMCTS Parameters:")  
    print(f"  MCTS Simulations: {args.numMCTSSims}")  
    print(f"  CPUCT: {args.cpuct}")  
      
    print("\nGame Parameters:")  
    print(f"  Board Size: {args.board_size}")  
      
    print("\nSystem Parameters:")  
    print(f"  CUDA Enabled: {args.cuda}")  
    print(f"  Checkpoint Directory: {args.checkpoint}")  
    print(f"  Load Model: {args.load_model}")  
    print(f"  Load Path: {args.load_folder_file}")  
      
    print("\nOptimization Parameters:")  
    print(f"  Prediction Batch Size: {args.prediction_batch_size}")  
    print(f"  Max MCTS States: {args.max_mcts_states}")  
    print(f"  Number of Workers: {args.num_workers}")  
    print(f"  Pin Memory: {args.pin_memory}")  
    print(f"  Async MCTS: {args.async_mcts}")  
    print(f"  Virtual Loss: {args.virtual_loss}")  
    print("==================\n")  
  
  
def main():  
    import argparse  
      
    parser = argparse.ArgumentParser()  
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")  
    parser.add_argument("--train", action="store_true")  
    parser.add_argument("--board_size", type=int, default=9)  
    # play arguments  
    parser.add_argument("--play", action="store_true")  
    parser.add_argument("--verbose", action="store_true")  
    parser.add_argument("--round", type=int, default=2)  
    parser.add_argument(  
        "--player1",  
        type=str,  
        default="human",  
        choices=["human", "random", "greedy", "alphazero"],  
    )  
    parser.add_argument(  
        "--player2",  
        type=str,  
        default="alphazero",  
        choices=["human", "random", "greedy", "alphazero"],  
    )  
    parser.add_argument("--ckpt_file", type=str, default="best.pth.tar")  
    parser.add_argument("--wandb", action="store_true", help="Use wandb to record the training process")  
    parser.add_argument("--wandb_project", type=str, default="alphazero-gomoku", help="wandb project name")  
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")  
    parser.add_argument("--wandb_id", type=str, default=None)  
      
    args_input = vars(parser.parse_args())  
      
    # Load config and override with command line arguments  
    args = load_config(args_input['config'])  
    for k, v in args_input.items():  
        if k != 'config':  
            args[k] = v  
      
    print_config(args)  
      
    g = game.GomokuGame(args.board_size)  
  
    if args.train:  
        # Initialize wandb  
        if args.wandb:  
            wandb.init(  
                project=args.wandb_project,  
                entity=args.wandb_entity,  
                config={  
                    "board_size": args.board_size,  
                    "num_iterations": args.numIters,  
                    "num_episodes": args.numEps,  
                    "num_mcts_sims": args.numMCTSSims,  
                    "batch_size": args.batch_size,  
                    "num_channels": args.num_channels,  
                    "learning_rate_min": args.min_lr,  
                    "learning_rate_max": args.max_lr,  
                    "grad_clip": args.grad_clip,  
                    "epochs": args.epochs,  
                    "dropout": args.dropout,  
                    "prediction_batch_size": args.prediction_batch_size,  
                    "max_mcts_states": args.max_mcts_states,  
                    "num_workers": args.num_workers,  
                    "pin_memory": args.pin_memory,  
                    "async_mcts": args.async_mcts,  
                    "virtual_loss": args.virtual_loss,  
                },  
                resume="allow"  
            )  
  
        nnet = NNetWrapper(g, args)  
        if args.load_model:  
            log.info(  
                'Loading checkpoint "%s/%s"...',  
                args.load_folder_file[0],  
                args.load_folder_file[1],  
            )  
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])  
  
        log.info("Loading the SelfCoach...")  
        s = SelfPlay(g, nnet, args)  
  
        log.info("Starting the learning process 🎉")  
        s.learn()  
  
    if args.play:  
        def getPlayFunc(name):  
            if name == "human":  
                return game.HumanGomokuPlayer(g).play  
            elif name == "random":  
                return game.RandomGomokuPlayer(g).play  
            elif name == "greedy":  
                return game.GreedyGomokuPlayer(g).play  
            elif name == "alphazero":  
                nnet = NNetWrapper(g, args)  
                nnet.load_checkpoint(args.checkpoint, args.ckpt_file)  
                mcts = MCTS(g, nnet, dotdict({  
                    "numMCTSSims": 800,   
                    "cpuct": 1.0,  
                    "prediction_batch_size": 32,  
                    "max_mcts_states": 100000,  
                    "cuda": args.cuda,  
                    "async_mcts": args.async_mcts,  
                    "virtual_loss": args.virtual_loss  
                }))  
                return lambda x: np.argmax(mcts.getActionProb(x, temp=0))  
            else:  
                raise ValueError("not support player name {}".format(name))  
  
        player1 = getPlayFunc(args.player1)  
        player2 = getPlayFunc(args.player2)  
          
        arena = game.Arena(player1, player2, g, display=g.display)  
        results = arena.playGames(args.round, verbose=args.verbose)  
        print("Final results: Player1 wins {}, Player2 wins {}, Draws {}".format(*results))  
  
  
if __name__ == "__main__":  
    main()