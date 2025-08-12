"""
File for decentralized SGA implementation
"""

import numpy as np
# To have the matplotlib without pausing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from math import *
import time
from tqdm import tqdm
from tools import *


class Orchestrator_SGA:
    """
    Orchestrator class for SGA that controls SGA agents
    """
    def __init__(self, G, char_t, char_a, Lt=1,  metric="RPT"):
        #2D array, adjacency graph of communication
        self.G = G
        # int, number of agents
        self.na = G.shape[0]
        # int, number of agents
        self.nt = len(char_t)
        # capacity per agent
        self.Lt = Lt

        # list of size 2, x limits of problem
        self.char_t = char_t
        # list of size 2, y limits of problem
        self.char_a = char_a
        # list of all agents
        self.agents = []
        # list of all tasks
        self.tasks = []

        # clock launch
        self.start_time = time.perf_counter()

        self.metric = metric

        # initialize tasks
        self.initialize_all()
        self.bid_history = []
        self.assig_history = []
        self.max_times = []
        self.all_times = [0 for _ in range(self.na)]

    def launch_agents(self):
        """
        SGA iterations to determine final allocation and total score
        :return: final allocation, minsum score, makespan score
        """
        Nmin = int(min(self.nt, self.Lt * self.na))
        for iter in tqdm(range(Nmin)):
            I = list(range(self.na))

            for i in I:
                t0 = time.time()
                self.agents[i].auto_assig()
                delta = time.time() - t0
                self.all_times[i] += delta

            for _ in range(len(self.agents)):
                for i in I:
                    t0 = time.time()
                    self.agents[i].consensus(self.agents)
                    delta = time.time() - t0
                    self.all_times[i] += delta

            for i in I:
                t0 = time.time()
                self.agents[i].resolve(iter)
                delta = time.time() - t0
                self.all_times[i] += delta

            assignment, bid, max_time = self.gather_info()
            self.assig_history.append(assignment)
            self.bid_history.append(bid)
            self.max_times.append(max_time)
        return self.assig_history[-1], self.bid_history[-1], self.max_times[-1]

    def gather_info(self):
        """
        At a given time (iteration) compute assignment, minsum and makespan
        :return: assignment list, minsum, makespan
        """
        bid_sum = 0
        assignment = []
        max_time = 0
        for i in range(self.na):
            agent = self.agents[i]
            a_time = agent.evaluate_path(agent.p, metric="result")
            if a_time > max_time:
                max_time = a_time
            bid_sum += a_time
            assignment.append(copy.deepcopy(agent.p))

        return assignment, np.round(bid_sum, 6), max_time

    def initialize_all(self):
        self.initialize_tasks()
        self.initialize_agents()

    def initialize_tasks(self):
        self.tasks = []
        for j in range(self.nt):
            char = self.char_t[j]
            self.tasks.append(Task(id=j, char=char))

    def initialize_agents(self):
        self.agents = []
        for i in range(self.na):
            char = self.char_a[i]
            self.agents.append(
                SGA_Agent(id=i, G=self.G, char=char, tasks=self.tasks, Lt=self.Lt, start_time=self.start_time, metric=self.metric))

    def print_perf(self):
        """
        Performance printer
        :return:
        """
        for i in range(self.na):
            path = self.assig_history[-1][i]
            score = self.agents[i].evaluate_path(path, metric="result")
            cvg_time = np.round(1000 * self.all_times[i])
            print("Agent {} converged: Path = {}, Score = {}, cvg time = {} ms".format(i, path, score, cvg_time))


class SGA_Agent:
    def __init__(self, id, G, char, tasks, Lt=2, start_time=0, metric="RPT"):
        # int, id of agent
        self.id = id
        # communication matrix G (symmetrical), size Na * Na
        self.G = G
        # int, nb of agents
        self.na = G.shape[0]
        # int, number of neighbors according to G
        self.nb_neigh = np.sum(self.G[id, :])
        # tuple, position in cartesian plane
        self.pos = np.array([char[0], char[1]])
        self.speed = char[2]
        # list of tasks (should be of size =nb agents)
        self.tasks = tasks
        # int, nb of tasks
        self.nt = len(tasks)
        # capacity
        self.Lt = Lt

        # path /ordered bundle
        self.p = []

        # remaining tasks
        self.J = []

        self.metric = metric
        if self.metric == "RPT":
            self.min_val = -10000
        elif metric == "TDR":
            self.min_val = 0
        for j in range(self.nt):
            self.J.append(j)
        self.init_J_len = len(self.J)

        self.Wa = []

        for j in self.J:
            marg = self.evaluate_path(self.p + [j], self.metric)
            self.Wa.append(marg)

        self.POS = [0 for _ in self.J]
        best_ind =np.where(np.array(self.Wa) == max(self.Wa))[0][0]

        self.j = self.J[best_ind]
        self.w = self.Wa[best_ind]

        self.placement = None

        self.astar = None
        self.jstar = None
        self.wstar = self.min_val

        self.previous_winner= True

        # clock start time
        self.start_time = start_time
        self.S = 0

    def evaluate_path(self, P, metric="RPT"):
        """
        Evaluate path P according to either the RPT metric or the TDR one. Result is the minsum metric (used only for final rendering)
        :param P: path = list of task indices
        :return: evaluation of path
        """
        current_pos = self.pos
        score = 0
        time = 0
        if len(P) > 0:
            for j in range(len(P)):
                task = self.tasks[P[j]]
                time += (np.linalg.norm(np.array(current_pos) - np.array(task.pos))) / self.speed
                time += task.duration
                if metric == "RPT" or metric == "result":
                    score -= time
                    time = 0
                elif metric == "TDR":
                    score += (task.lamb) ** time
                current_pos = task.pos
        if metric == "result":
            return -score
        else:
            return score

    def calc_marg_list(self, p, J):
        """
        Determines the task of optimal marginal gain in J
        :param p: assigned path
        :param J: remaining tasks
        :return: best marginal gain, corresponding task, position of insertion in path
        """
        Wa = []
        POS = []
        for j in J:
            best_marg, j_best_pos = self.calc_marg_task(p, j)
            Wa.append(best_marg)
            POS.append(j_best_pos)
        self.Wa = Wa
        self.POS = POS
        wa_star = max(self.Wa)
        best_ind = np.argwhere(np.array(self.Wa) == wa_star)[0][0]
        pos_a = POS[best_ind]
        ja_star = J[best_ind]
        return wa_star, ja_star, pos_a


    def calc_marg_task(self, p, j):
        """
        Determines the marginal gain of j
        :param p: assigned path
        :param j: task to evaluate the marginal gain
        :return: marginal gain of j
        """
        marg_pos = []
        for pos in range(len(p) + 1):
            d_path = copy.deepcopy(p)
            d_path.insert(pos, j)
            if self.metric == "RPT":
                marg = self.evaluate_path(d_path, self.metric)
            elif self.metric == "TDR":
                marg = self.evaluate_path(d_path, self.metric) - self.S
            marg_pos.append(marg)
        best_marg = max(marg_pos)
        j_best_pos = np.argwhere(np.array(marg_pos) == best_marg)[-1][0]
        return best_marg, j_best_pos

    def auto_assig(self):
        """
        Auto-assignment (of 1 task) function
        :return:
        """
        if len(self.p) < self.Lt and len(self.J)>0:
            if self.previous_winner == False:
                self.w = max(self.Wa)
                ind = self.Wa.index(self.w)
                self.j = self.J[ind]
                self.placement = self.POS[ind]

                self.astar = self.id
                self.jstar = self.j
                self.wstar = self.w
                return
            w, j, place = self.calc_marg_list(self.p, self.J)
            self.j = j
            self.w = w
            self.placement = place

            self.astar = self.id
            self.jstar = j
            self.wstar = w

        else:
            self.wstar = self.min_val
            self.jstar = None
            self.astar = None



    def consensus(self, all_agents):
        """
        Consensus among all the agents to determine who has the best bid (marginal gain)
        :param all_agents: list of all agents
        :return:
        """
        neigh_index = np.argwhere(self.G[self.id, :] == 1).flatten()
        for k in neigh_index:
            neigh = all_agents[k]
            if neigh.wstar > self.wstar:
                self.astar = neigh.astar
                self.jstar = neigh.jstar
                self.wstar = neigh.wstar
            if self.astar is None:
                self.astar = neigh.astar
                self.jstar = neigh.jstar
                self.wstar = neigh.wstar
            elif neigh.wstar == self.wstar and neigh.astar < self.astar:
                self.astar = neigh.astar
                self.jstar = neigh.jstar
                self.wstar = neigh.wstar

    def resolve(self, iter):
        """
        The agent with best bid (determined from consensus) inserts its task. The other remove this task from J
        :param iter:
        :return:
        """
        if len(self.p) < self.Lt:
            if self.astar == self.id:
                self.p.insert(self.placement, self.jstar)
                ind =self.J.index(self.jstar)
                self.J.remove(self.jstar)
                self.Wa.pop(ind)
                self.POS.pop(ind)
                self.previous_winner = True
                self.S = self.evaluate_path(self.p, self.metric)
            else:
                if self.jstar in self.J:
                    ind =self.J.index(self.jstar)
                    self.J.remove(self.jstar)
                    self.Wa.pop(ind)
                    self.POS.pop(ind)
                self.previous_winner = False


if __name__ == "__main__":
    # On seed 5, CBBA and GCBBA (SGA) allocations are different for RPT
    seed = 5
    np.random.seed(seed)
    na = 10
    nt = 100
    Lt = ceil(nt / na)
    xlim = [-5, 5]
    ylim = xlim
    sp_lim = [1, 5]
    dur_lim = [1, 5]
    metric = "RPT"

    #communication graph initialization
    raw_graph, G = create_graph(na, graph_type="full", seed=seed)
    agents, tasks = task_agent_init(na=na, nt=nt, pos_lim=xlim, sp_lim=sp_lim, dur_lim=dur_lim, lamb_lim=[0.95, 0.95],
                                    clim=[1, 1])
    orch_sga = Orchestrator_SGA(G, tasks, agents, Lt, metric=metric)

    #allocation launching
    t0 = time.time()
    assig, tot_score, makespan = orch_sga.launch_agents()
    tf0 = np.round(1000 * (time.time() - t0))

    print("SGA-{} Min-Sum = {}; makespan = {};  assignment = {}".format(metric,tot_score, makespan,
                                                                                              assig))
    draw_paths(tasks, agents, assig, tot_score, title="SGA-{}".format(metric))





