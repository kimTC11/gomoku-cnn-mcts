import logging  
import math  
import os  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from tqdm import tqdm  
from collections import deque  
from random import shuffle  
import wandb  
import yaml  
  
import 5x5.game_4ch as game  
  
logging.basicConfig(level=logging.INFO)  
log = logging.getLogger(__name__)  
  
  
class MCTS:  
    """  
    This class handles the MCTS tree.  
    """  
  
    def __init__(self, game, nnet, args):  
        self.game = game  
        self.nnet = nnet  
        self.args = args  
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)  
        self.Nsa = {}  # stores #times edge s,a was visited  
        self.Ns = {}  # stores #times board s was visited  
        self.Ps = {}  # stores initial policy (returned by neural net)  
  
        self.Es = {}  # stores game.getGameEnded for board s  
        self.Vs = {}  # stores game.getValidMoves for board s  
  
    def getActionProb(self, canonicalBoard, temp=1):  
        """  
        This function performs numMCTSSims simulations of MCTS starting from  
        canonicalBoard.  
  
        Returns:  
            probs: a policy vector where the probability of the ith action is  
                   proportional to Nsa[(s,a)]**(1./temp)  
        """  
        for _ in range(self.args.numMCTSSims):  
            self.search(canonicalBoard)  
  
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
        This function performs one iteration of MCTS. It is recursively called  
        till a leaf node is found. The action chosen at each node is one that  
        has the maximum upper confidence bound as in the paper.  
  
        Once a leaf node is found, the neural network is called to return an  
        initial policy P and a value v for the state. This value is propagated  
        up the search path. In case the leaf node is a terminal state, the  
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are  
        updated.  
  
        NOTE: Since v is in [-1,1] and if v is the value of a  
        state for the current player, then its value is -v for the other player.  
  
        Returns:  
            v: the value of the current canonicalBoard  
        """  
  
        s = self.game.stringRepresentation(canonicalBoard)  
  
        if s not in self.Es:  
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)  
        if self.Es[s] is not None:  
            # terminal node  
            return self.Es[s]  
  
        if s not in self.Ps:  
            # leaf node  
            self.Ps[s], v = self.nnet.predict(canonicalBoard)  
            valids = self.game.getValidMoves(canonicalBoard, 1)  
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves  
            sum_Ps_s = np.sum(self.Ps[s])  
            if sum_Ps_s > 0:  
                self.Ps[s] /= sum_Ps_s  # renormalize  
            else:  
                # if all valid moves were masked make all valid moves equally probable  
                log.error("All valid moves were masked, doing a workaround.")  
                self.Ps[s] = self.Ps[s] + valids  
                self.Ps[s] /= np.sum(self.Ps[s])  
  
            self.Vs[s] = valids  
            self.Ns[s] = 0  
            return v  
  
        valids = self.Vs[s]  
        cur_best = -float("inf")  
        best_act = -1  
  
        # pick the action with the highest upper confidence bound  
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
  
  
class ResidualBlock(nn.Module):  
    """  
    Residual block implementation inspired by SLAP codebase.  
    Source: https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/  
    """  
    def __init__(self, in_channels, out_channels, stride=[1, 1], downsample=None):  
        """  
        A basic residual block of ResNet, allow downsampling of features.  
          
        Args:  
            in_channels: Number of channels that the input have  
            out_channels: Number of channels that the output have  
            stride: strides in convolutional layers  
            downsample: A callable to be applied before addition of residual mapping  
        """  
        super(ResidualBlock, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,   
                               stride=stride[0], padding=1, bias=False)  
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,   
                               stride=stride[1], padding=1, bias=False)  
        self.bn = nn.BatchNorm2d(out_channels)  
        self.downsample = downsample  
  
    def forward(self, x):  
        residual = x  
        # applying a downsample function before adding it to the output  
        if self.downsample is not None:  
            residual = self.downsample(residual)  
        out = F.relu(self.bn(self.conv1(x)))  
        out = self.bn(self.conv2(out))  
        # note that adding residual before activation   
        out = out + residual  
        out = F.relu(out)  
        return out  
  
  
class GomokuNNet(nn.Module):  
    """  
    Standard CNN policy-value network without residual blocks.  
    """  
    def __init__(self, game, args):  
        # game params  
        self.board_x, self.board_y = game.getBoardSize()  
        self.action_size = game.getActionSize()  
        self.args = args  
  
        super(GomokuNNet, self).__init__()  
        # Fixed: Changed from 1 to 4 input channels for proper state representation  
        self.conv1 = nn.Conv2d(4, args.num_channels, 3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)  
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
        # s: batch_size x 4 x board_x x board_y  
        s = s.view(-1, 4, self.board_x, self.board_y)  
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
  
  
class GomokuResNet(nn.Module):  
    """  
    ResNet-based policy-value network with residual blocks.  
    Inspired by SLAP codebase implementation.  
    """  
    def __init__(self, game, args):  
        super(GomokuResNet, self).__init__()  
        self.board_x, self.board_y = game.getBoardSize()  
        self.action_size = game.getActionSize()  
        self.args = args  
        self.num_res_blocks = getattr(args, 'num_res_blocks', 3)  
          
        # Common layers  
        self.conv1 = nn.Conv2d(4, args.num_channels, kernel_size=3, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(args.num_channels)  
          
        # Residual blocks  
        self.res_blocks = nn.ModuleList([  
            ResidualBlock(args.num_channels, args.num_channels) for _ in range(self.num_res_blocks)  
        ])  
          
        # Policy head  
        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1)  
        self.policy_bn = nn.BatchNorm2d(2)  
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)  
          
        # Value head  
        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1)  
        self.value_bn = nn.BatchNorm2d(1)  
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, 256)  
        self.value_fc2 = nn.Linear(256, 1)  
          
        self.dropout = nn.Dropout(args.dropout)  
  
    def forward(self, s):  
        # s: batch_size x 4 x board_x x board_y  
        s = s.view(-1, 4, self.board_x, self.board_y)  
          
        # Common layers  
        x = F.relu(self.bn1(self.conv1(s)))  
          
        # Residual blocks  
        for res_block in self.res_blocks:  
            x = res_block(x)  
          
        # Policy head  
        policy = F.relu(self.policy_bn(self.policy_conv(x)))  
        policy = self.dropout(policy.view(-1, 2 * self.board_x * self.board_y))  
        policy = self.policy_fc(policy)  
        policy = F.log_softmax(policy, dim=1)  
          
        # Value head  
        value = F.relu(self.value_bn(self.value_conv(x)))  
        value = self.dropout(value.view(-1, self.board_x * self.board_y))  
        value = self.dropout(F.relu(self.value_fc1(value)))  
        value = torch.tanh(self.value_fc2(value))  
          
        return policy, value  
  
  
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
    def __init__(self, game, args):  
        # Choose network architecture based on configuration  
        use_resnet = getattr(args, 'use_resnet', False)  
        if use_resnet:  
            self.nnet = GomokuResNet(game, args)  
        else:  
            self.nnet = GomokuNNet(game, args)  
              
        self.board_x, self.board_y = game.getBoardSize()  
        self.action_size = game.getActionSize()  
        self.args = args  
  
        # Device selection  
        if args.cuda:  
            self.nnet.cuda()  
        elif getattr(args, 'mps', False):  
            self.nnet.to('mps')  
          
        # Initialize optimizer  
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.max_lr)  
          
        # 1cycle learning rate parameters - will be calculated dynamically  
        self.current_step = 0  
        self.total_steps = 0  
  
    def get_learning_rate(self):  
        """Implement 1cycle learning rate strategy"""  
        if self.total_steps == 0 or self.current_step >= self.total_steps:  
            return self.args.min_lr  
          
        half_cycle = self.total_steps // 2  
          
        if self.current_step <= half_cycle:  
            # First phase: increase from min_lr to max_lr  
            phase = self.current_step / half_cycle  
            lr = self.args.min_lr + (self.args.max_lr - self.args.min_lr) * phase  
        else:  
            # Second phase: decrease from max_lr to min_lr  
            phase = (self.current_step - half_cycle) / half_cycle  
            lr = self.args.max_lr - (self.args.max_lr - self.args.min_lr) * phase  
          
        return lr  
  
    def train(self, examples):  
        """  
        examples: list of examples, each example is of form (board, pi, v)  
        """  
        # Calculate total_steps based on actual buffer size  
        batch_count = int(len(examples) / self.args.batch_size)  
        self.total_steps = self.args.epochs * batch_count  
          
        for epoch in range(self.args.epochs):  
            print("EPOCH ::: " + str(epoch + 1))  
            self.nnet.train()  
            pi_losses = AverageMeter()  
            v_losses = AverageMeter()  
  
            t = tqdm(range(batch_count), desc="Training Net")  
            for _ in t:  
                # Update learning rate  
                lr = self.get_learning_rate()  
                for param_group in self.optimizer.param_groups:  
                    param_group['lr'] = lr  
                self.current_step += 1  
  
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)  
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))  
                boards = torch.FloatTensor(np.array(boards).astype(np.float32))  
                target_pis = torch.FloatTensor(np.array(pis))  
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))  
  
                # Move to device  
                if self.args.cuda:  
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()  
                elif getattr(self.args, 'mps', False):  
                    boards, target_pis, target_vs = boards.to('mps'), target_pis.to('mps'), target_vs.to('mps')  
  
                # compute output  
                out_pi, out_v = self.nnet(boards)  
                l_pi = self.loss_pi(target_pis, out_pi)  
                l_v = self.loss_v(target_vs, out_v)  
                total_loss = l_pi + l_v  
  
                # record loss  
                pi_losses.update(l_pi.item(), boards.size(0))  
                v_losses.update(l_v.item(), boards.size(0))  
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, lr=f"{lr:.1e}")  
  
                # compute gradient and do SGD step  
                self.optimizer.zero_grad()  
                total_loss.backward()  
                  
                # Fixed: Add gradient clipping with proper condition  
                if self.args.grad_clip > 0:  
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
        board: np array with 4-channel board (4, board_x, board_y)
        """  
        # preparing input  
        board = torch.FloatTensor(board.astype(np.float32))  
        if self.args.cuda:  
            board = board.cuda()  
        elif getattr(self.args, 'mps', False):  
            board = board.to('mps')  
        # Handle 4-channel input: (4, board_x, board_y) -> (1, 4, board_x, board_y)
        if board.ndim == 3:  # (4, H, W)
            board = board.unsqueeze(0)  # (1, 4, H, W)
        elif board.ndim == 2:  # Legacy (H, W) - shouldn't happen with 4-channel
            board = board.view(1, 1, self.board_x, self.board_y)  # (1, 1, H, W)  
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
            print(f"Checkpoint Directory does not exist! Making directory {folder}")  
            os.mkdir(folder)  
        else:  
            print("Checkpoint Directory exists!")  
          
        # Fixed: Save optimizer state and current_step for proper resumption  
        torch.save({  
            "state_dict": self.nnet.state_dict(),  
            "optimizer": self.optimizer.state_dict(),  
            "current_step": self.current_step,  
        }, filepath)  
  
    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):  
        folder = folder.rstrip('/')  
        filepath = os.path.join(folder, filename)  
        if not os.path.exists(filepath):  
            raise ValueError(f"No model in path {filepath}")  
          
        # Fixed: Support CPU loading  
        if self.args.cuda:  
            map_location = None  
        elif getattr(self.args, 'mps', False):  
            map_location = 'mps'  
        else:  
            map_location = 'cpu'  
          
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)  
        self.nnet.load_state_dict(checkpoint["state_dict"])  
          
        # Load optimizer state and current_step if available  
        if "optimizer" in checkpoint:  
            self.optimizer.load_state_dict(checkpoint["optimizer"])  
        if "current_step" in checkpoint:  
            self.current_step = checkpoint["current_step"]  
  
  
class SelfPlay:  
    """  
    This class executes the self-play + learning.  
    """  
  
    def __init__(self, game, nnet, args):  
        self.game = game  
        self.nnet = nnet  
        self.pnet = self.nnet.__class__(self.game, args)  # the competitor network  
        self.args = args  
        self.mcts = MCTS(self.game, self.nnet, self.args)  
        self.trainExamplesHistory = []  
        self.best_win_ratio = 0.0  # Track best win ratio  
  
    def executeEpisode(self):  
        """  
        This function executes one episode of self-play, starting with player 1.  
        As the game is played, each turn is added as a training example to  
        trainExamples. The game is played till the game ends. After the game  
        ends, the outcome of the game is used to assign values to each example  
        in trainExamples.  
  
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter  
        uses temp=0.  
  
        Returns:  
            trainExamples: a list of examples of the form (canonicalBoard, pi, v)  
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
            board, self.curPlayer = self.game.getNextState(  
                board, self.curPlayer, action  
            )  
  
            r = self.game.getGameEnded(board, self.curPlayer)  
  
            if r is not None:  
                return [  
                    (x[0], x[2], r * (1 if self.curPlayer == x[1] else -1))  
                    for x in trainExamples  
                ]  
  
    def learn(self):  
        """  
        Performs numIters iterations with numEps episodes of self-play in each  
        iteration. After every iteration, it retrains neural network with  
        examples in trainExamples (which has a maximum length of maxlenofQueue).  
        It then pits the new neural network against the best one and accepts it  
        only if it wins >= updateThreshold fraction of games.  
        """  
  
        for i in range(1, self.args.numIters + 1):  
            log.info(f"Starting Iter #{i} ...")  
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)  
  
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):  
                self.mcts = MCTS(self.game, self.nnet, self.args)  
                iterationTrainExamples += self.executeEpisode()  
  
            self.trainExamplesHistory.append(iterationTrainExamples)  
  
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:  
                log.warning(f"Removing oldest entry. len={len(self.trainExamplesHistory)}")  
                self.trainExamplesHistory.pop(0)  
  
            # shuffle examples before training  
            trainExamples = []  
            for e in self.trainExamplesHistory:  
                trainExamples.extend(e)  
            shuffle(trainExamples)  
  
            # Save current model as temp  
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")  
              
            # Fixed: Load best model (or temp if no best exists yet)  
            best_path = os.path.join(self.args.checkpoint, "best.pth.tar")  
            if os.path.exists(best_path):  
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")  
            else:  
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")  
              
            pmcts = MCTS(self.game, self.pnet, self.args)  
  
            # Train new model  
            self.nnet.train(trainExamples)  
            nmcts = MCTS(self.game, self.nnet, self.args)  
  
            log.info("PITTING AGAINST BEST VERSION")  
            arena = game.Arena(  
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),  
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),  
                self.game,  
            )  
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)  
  
            log.info("NEW/BEST WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))  
              
            # Fixed: Calculate win ratio properly  
            win_ratio = float(nwins) / (pwins + nwins) if (pwins + nwins) > 0 else 0  
              
            if win_ratio >= self.args.updateThreshold:  
                log.info(f"ACCEPTING NEW MODEL (win ratio: {win_ratio:.3f})")  
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")  
                self.best_win_ratio = win_ratio  
                  
                # Log to wandb if enabled  
                if getattr(self.args, 'wandb', False):  
                    wandb.log({  
                        'iteration': i,  
                        'win_ratio': win_ratio,  
                        'accepted': True,  
                        'pwins': pwins,  
                        'nwins': nwins,  
                        'draws': draws,  
                    })  
            else:  
                log.info(f"REJECTING NEW MODEL (win ratio: {win_ratio:.3f} < {self.args.updateThreshold})")  
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")  
                  
                # Log to wandb if enabled  
                if getattr(self.args, 'wandb', False):  
                    wandb.log({  
                        'iteration': i,  
                        'win_ratio': win_ratio,  
                        'accepted': False,  
                        'pwins': pwins,  
                        'nwins': nwins,  
                        'draws': draws,  
                    })  
  
  
class dotdict(dict):  
    def __getattr__(self, name):  
        return self[name]  
  
  
def load_config(config_path):  
    with open(config_path, 'r') as f:  
        config = yaml.safe_load(f)  
      
    args = dotdict({})  
      
    # Training params  
    args.epochs = config['training']['epochs']  
    args.batch_size = config['training']['batch_size']  
    args.numIters = config['training']['num_iterations']  
    args.numEps = config['training']['num_episodes']  
    args.maxlenOfQueue = config['training']['max_queue_length']  
    args.numItersForTrainExamplesHistory = config['training']['num_iters_history']  
    args.updateThreshold = config['training']['update_threshold']  
    args.arenaCompare = config['training']['arena_compare']  
    args.tempThreshold = config['training']['temp_threshold']  
      
    # Network params  
    args.num_channels = config['network']['num_channels']  
    args.dropout = config['network']['dropout']  
    args.min_lr = config['network']['learning_rate']['min']  
    args.max_lr = config['network']['learning_rate']['max']  
    args.grad_clip = config['network']['grad_clip']  
    args.use_resnet = config['network'].get('use_resnet', False)  
    args.num_res_blocks = config['network'].get('num_res_blocks', 3)  
      
    # MCTS params  
    args.numMCTSSims = config['mcts']['num_sims']  
    args.cpuct = config['mcts']['cpuct']  
      
    # Game params  
    args.board_size = config['game']['board_size']  
      
    # System params  
    args.cuda = config['system']['cuda'] and torch.cuda.is_available()  
    args.mps = config['system'].get('mps', True) and torch.backends.mps.is_available()  
    args.device = 'cuda' if args.cuda else ('mps' if args.mps else 'cpu')  
    args.checkpoint = config['system']['checkpoint_dir']  
    args.load_model = config['system']['load_model']  
    args.load_folder_file = tuple(config['system']['load_folder_file'])  
      
    return args  
  
  
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
    print(f"  Use ResNet: {getattr(args, 'use_resnet', False)}")  
    if getattr(args, 'use_resnet', False):  
        print(f"  Number of Residual Blocks: {getattr(args, 'num_res_blocks', 3)}")  
      
    print("\nMCTS Parameters:")  
    print(f"  MCTS Simulations: {args.numMCTSSims}")  
    print(f"  CPUCT: {args.cpuct}")  
      
    print("\nGame Parameters:")  
    print(f"  Board Size: {args.board_size}")  
      
    print("\nSystem Parameters:")  
    print(f"  CUDA Enabled: {args.cuda}")  
    print(f"  MPS Enabled: {getattr(args, 'mps', False)}")  
    print(f"  Device: {args.device}")  
    print(f"  Checkpoint Directory: {args.checkpoint}")  
    print(f"  Load Model: {args.load_model}")  
    print(f"  Load Path: {args.load_folder_file}")  
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
      
    # Print configuration  
    print_config(args)  
      
    g = game.GomokuGame(args.board_size)  
  
    if args.train:  
        # Initialize wandb  
        if args.wandb:  
            wandb_config = {  
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
                "use_resnet": getattr(args, 'use_resnet', False),  
            }  
            if getattr(args, 'use_resnet', False):  
                wandb_config["num_res_blocks"] = getattr(args, 'num_res_blocks', 3)  
              
            wandb.init(  
                project=args.wandb_project,  
                entity=args.wandb_entity,  
                config=wandb_config,  
                resume="allow",  
                id=args.wandb_id  
            )  
  
        nnet = NNetWrapper(g, args)  
        if args.load_model:  
            log.info(  
                'Loading checkpoint "%s/%s"...',  
                args.load_folder_file[0],  
                args.load_folder_file[1],  
            )  
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])  
  
        log.info("Loading the SelfPlay...")  
        s = SelfPlay(g, nnet, args)  
  
        log.info("Starting the learning process 🎉")  
        s.learn()  
          
        # Close wandb  
        if args.wandb:  
            wandb.finish()  
  
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
                mcts = MCTS(g, nnet, dotdict({"numMCTSSims": 800, "cpuct": 1.0}))  
                return lambda x: np.argmax(mcts.getActionProb(x, temp=0))  
            else:  
                raise ValueError(f"not support player name {name}")  
  
        player1 = getPlayFunc(args.player1)  
        player2 = getPlayFunc(args.player2)  
          
        arena = game.Arena(player1, player2, g, display=g.display)  
        results = arena.playGames(args.round, verbose=args.verbose)  
        print("Final results: Player1 wins {}, Player2 wins {}, Draws {}".format(*results))  
  
  
if __name__ == "__main__":  
    main()