import networkx as nx
import numpy as np
# To have the matplotlib without pausing
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from math import *
import time
from tqdm import tqdm
from tools import *



class Orchestrator_CBBA:
    """
    CBBA Orchestrator controlling CBBA agents 
    """
    def __init__(self, G, D, char_t, char_a, Lt=1, metric = "RPT"):
        self.G = G
        # aint, number of agents
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

        self.D = D

        # initialize tasks
        self.initialize_all()
        self.bid_history = []
        self.assig_history = []
        self.max_times = []
        self.all_times = [0 for _ in range(self.na)]

        self.cvg_iter = self.nt

    #merged
    def launch_agents(self, method = "baseline", detector = "none"):
        """
        CBBA iterations to determine assignment, minsum and makespan
        :param method: "baseline" => gives baseline CBBA allocation. "global" => gives GCBBA (ours) allocation
        :param detector: "none" => no convergence detector. 
        "centralized" => centralized detector suited to CBBA. 
        "decentralized" => our decentralized detector suited to GCBBA.
        :return: allocation, minsum, makespan
        """
        # track progress
        D = self.D
        Nmin = int(min(self.nt, self.Lt * self.na))
        if method == "baseline":
            nb_iter = Nmin*D
            nb_cons = 1
        elif method == "global":
            nb_iter = Nmin
            nb_cons = 2*D
        else:
            nb_iter = Nmin
            nb_cons = 2*D
        build_bundle = "FULLBUNDLE" if method == "baseline" else "ADD"
        for iter in tqdm(range(nb_iter)):
            I = list(range(self.na))
            for i in I:
                if detector == "decentralized" and self.agents[i].converged==False:
                    self.agents[i].create_bundle(iter, build_bundle = build_bundle)
                elif detector != "decentralized":
                    self.agents[i].create_bundle(iter, build_bundle=build_bundle)
            for _ in range(nb_cons):
                all_agents = copy.deepcopy(self.agents)
                # last consensus iteration ?
                if _ == nb_cons-1:
                    index = "last"
                else:
                    index = "else"
                for i in I:
                    if detector == "decentralized" and self.agents[i].converged ==False:
                        self.agents[i].resolve_conflict(all_agents, iter = nb_cons * iter + _, index = index)
                    elif detector != "decentralized":
                        self.agents[i].resolve_conflict(all_agents, iter = nb_cons * iter + _, index = index)
            assignment, bid, max_time = self.gather_info()
            self.assig_history.append(assignment)
            self.bid_history.append(bid)
            self.max_times.append(max_time)

            if detector == "centralized":
                if check_conflict(self.assig_history[-1], self.nt):
                    pass
                else:
                    print("EARLY CONVERGENCE AT {}/{}".format(iter+1, nb_iter))
                    self.cvg_iter = iter
                    break

            if detector == "decentralized":
                if np.prod([self.agents[i].converged for i in range(1)]) == False:
                    pass
                else:
                    print("EARLY CONVERGENCE AT {}/{}".format(iter+1, nb_iter))
                    self.cvg_iter = iter
                    break
        return self.assig_history[-1], self.bid_history[-1], self.max_times[-1]

    def get_cvg_iter(self):
        return self.cvg_iter

    def gather_info(self):
        """
        At a given time (iteration) compute assignment, minsum and makespan
        :return: assignment list, minsum and makespan
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
                CBBA_Agent(id=i, G=self.G, char=char, tasks=self.tasks, Lt=self.Lt, start_time=self.start_time,
                           metric=self.metric, D = self.D))

    def print_perf(self):
        """
        Performance printing function
        :return:
        """
        for i in range(self.na):
            path = self.assig_history[-1][i]
            score = self.agents[i].evaluate_path(path, metric="RPT")
            cvg_time = np.round(1000 * self.all_times[i])
            print("Agent {} converged: Path = {}, Score = {}, cvg time = {} ms".format(i, path, score, cvg_time))
            
    def compute_TDR_obj(self):
        """
        compute TDR sum value of final allocation (THE HIGHER THE BETTER)
        :return: TDR sum value
        """
        S = 0
        lamb = 0.95
        for i in range(self.na):
            agent = self.agents[i]
            P = agent.p
            for j in range(len(agent.p)):
                S += self.tasks[P[j]].cbar * lamb**(agent.evaluate_path(P[:(j+1)], metric = "result"))
        return S

    def compute_RPT_obj(self):
        """
        compute the effective RPT sum value of final allocation (THE LOWER THE BETTER)
        :return: effective RPT sum value
        """
        S = 0
        for i in range(self.na):
            agent = self.agents[i]
            P = agent.p
            B = agent.b
            J = []
            for j in range(len(B)):
                J.append(B[j])
                Pb = copy.deepcopy(P)
                for k in P:
                    if k not in J:
                        Pb.remove(k)
                S += abs(agent.evaluate_path(Pb, metric="RPT"))
        return S



class CBBA_Agent:

    def __init__(self, id, G, char, tasks, Lt=2, start_time=0, metric="RPT", D=1):
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
        self.D = D
        self.makespan = 0
        self.updated_makespan = 0
        self.length = 0


        self.metric = metric
        if self.metric == "RPT":
            self.min_val = -1e20
        elif metric == "TDR":
            self.min_val = 0

        # list of winning bids for each task (size Nt)
        self.y = [self.min_val for _ in range(self.nt)]
        # list of winners for each task (size Nt)
        # self.z = np.array([None for _ in range(self.nt)]).reshape((1,-1))
        self.z = [None for _ in range(self.nt)]
        self.z_before = [None for _ in range(self.nt)]
        self.z_pre_consensus = [None for _ in range(self.nt)]
        # list of potentially winning bids (size Nt)
        self.h = [self.min_val for _ in range(self.nt)]
        # list of bids on each task (size Nt)
        self.c = [self.min_val for _ in range(self.nt)]

        self.j_outbid = False
        self.outbidder = None
        self.outbid_pos = None
        # bundle
        self.b = []
        # path /ordered bundle
        self.p = []

        # timestamps
        self.s = [-inf for _ in range(self.na)]
        self.s[self.id] = 0
        # tasks not in path / bundle
        self.filtered_index = list(range(self.nt))
        # clock start time
        self.start_time = start_time
        self.S = [0]
        self.converged = False
        #observation list of other agents convergence
        self.their_net_cvg = [False for _ in range(self.D)]
        self.cvg_counter = 0

        #marginal gain list
        self.Wa = []
        #task insert position list
        self.placement = []
        #agent has won at previous iteration ? (optional)
        self.flag_won = True
        #size of path at previous iteration
        self.len_p_before = 0



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


    def compute_c(self, j, metric="RPT"):
        """
        Compute marginal gain c[j] of j and optimal position in path
        :param j: task j, assumed to not be in bundle
        :return: c[j], optimal position (index) of j in the path
        """
        # self.filtered_index = [k for k,task in enumerate(self.tasks) if task.id not in self.p]
        path_bids = []
        placements = []
        P = self.p
        cur_task = self.tasks[j].id
        for pos in range(len(self.p) + 1):
            P1 = copy.deepcopy(P)
            P1.insert(pos, cur_task)
            if metric == "TDR":
                mg = self.evaluate_path(P1, metric) - self.S[-1]
            elif metric == "RPT":
                mg = self.evaluate_path(P1, metric)
            path_bids.append(mg)
            placements.append(pos)
        c = np.max(path_bids)
        optimal_pos = np.argwhere(path_bids == c)[-1][0]
        return c, optimal_pos

    def create_bundle(self, iter, build_bundle = "ADD"):
        """
        Path building phase for each agent
        :param iter:
        :param build_bundle: "ADD" is the method of GCBBA (adds only one task). "FULLBUNDLE" is baseline CBBA's method (builds full path)
        :return:
        """
        self.updated_makespan = self.makespan

        if build_bundle == "FULLBUNDLE":
            while len(self.b) < self.Lt:
                optimal_placement = np.zeros(self.nt)
                self.filtered_index = [k for k, task in enumerate(self.tasks) if task.id not in self.p]
                for j in self.filtered_index:
                    c, opt_place = self.compute_c(j, metric = self.metric)
                    self.c[j] = c
                    optimal_placement[j] = opt_place
                if self.metric == "TDR":
                    cbar = copy.deepcopy(self.c)
                    if len(self.b)>0:
                        min_c = np.min([self.c[j] for j in self.b])
                        for j in self.filtered_index:
                            cbar[j] = min(self.c[j],min_c)
                    bids = [self.c[j] if (cbar[j] > self.y[j] and j in self.filtered_index or cbar[j] == self.y[
                        j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                            range(self.nt)]
                    J = np.argmax(bids)
                    self.c = cbar
                else:
                    bids = [self.c[j] if (self.c[j] > self.y[j] and j in self.filtered_index or self.c[j] == self.y[j] and j in self.filtered_index and self.z[j]>self.id) else self.min_val for j in
                            range(self.nt)]
                    J = np.argmax(bids)
                if J in self.p or (bids[J] <= self.min_val):
                    return

                best_task = self.tasks[J].id
                self.b.append(best_task)
                self.p.insert(int(optimal_placement[J]), best_task)
                self.S.append(self.evaluate_path(self.p, self.metric))
                self.y[J] = self.c[J]
                self.z[J] = self.id
        #build_bundle = add
        else:
            if len(self.b) < self.Lt:
                self.filtered_index = [k for k, task in enumerate(self.tasks) if task.id not in self.p]
                if self.flag_won==True:
                    self.placement = np.zeros(self.nt)
                    for j in self.filtered_index:
                        c, opt_place = self.compute_c(j, metric=self.metric)
                        self.c[j] = c
                        self.placement[j] = opt_place
                    if self.metric == "TDR":
                        cbar = copy.deepcopy(self.c)
                        if len(self.b)>0:
                            min_c = np.min([self.c[j] for j in self.b])
                            for j in self.filtered_index:
                                #cbar[j] = min(self.c[j],min_c)
                                cbar[j] = self.c[j]
                        bids = [self.c[j] if (cbar[j] > self.y[j] and j in self.filtered_index or cbar[j] == self.y[
                            j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                                range(self.nt)]
                        J = np.argmax(bids)
                        self.c = cbar
                    else:
                        bids = [self.c[j] if (self.c[j] > self.y[j] and j in self.filtered_index or self.c[j] == self.y[
                            j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                                range(self.nt)]
                        J = np.argmax(bids)
                else:
                    bids = [self.c[j] if (self.c[j] > self.y[j] and j in self.filtered_index or self.c[j] == self.y[
                        j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                            range(self.nt)]
                    J = np.argmax(bids)

                if J in self.p or (bids[J] <= self.min_val):
                    return

                best_task = self.tasks[J].id
                self.b.append(best_task)
                self.p.insert(int(self.placement[J]), best_task)
                self.S.append(self.evaluate_path(self.p, self.metric))
                self.y[J] = self.c[J]
                self.z[J] = self.id
                self.updated_makespan = max(self.updated_makespan, -self.evaluate_path(self.p, "RPT"))

    def compute_s(self, neigh, iter):
        """
        compute the new timestamps based on communication (reception) with agent neigh
        :param neigh: agent object, assumed to be neighbor of self
        :param iter: the timestamp updated alue is the current iteration of algo (can also be based on clock time -> current time - start time)
        :return:
        """
        cur_time = time.perf_counter()
        diff = cur_time - self.start_time
        self.s[self.id] = iter
        self.s[neigh.id] = iter
        #self.s[neigh.id] = diff
        not_neigh_index = np.argwhere(self.G[self.id, :] == 0).flatten()
        greater_index = np.argwhere(np.array(neigh.s) > np.array(self.s)).flatten()
        intersect = list(set(greater_index).intersection(set(not_neigh_index)))
        self.s = [neigh.s[i] if i in intersect else self.s[i] for i in range(self.na)]


    def update(self, neigh, j):
        """
        update procedure
        :param neigh: agent object, assumed to be neighbor of self
        :param j: task j
        :return:
        """
        self.y[j] = neigh.y[j]
        self.z[j] = neigh.z[j]

        bundle = self.b
        if j in bundle:
            self.flag_won = False
            bundle_index = bundle.index(j)
            tasks_to_remove = self.b[bundle_index:]

            self.y = [self.y[i] if i not in tasks_to_remove else self.min_val for i in range(self.nt)]
            self.z = [self.z[i] if i not in tasks_to_remove else None for i in range(self.nt)]
            self.y[j] = neigh.y[j]
            self.z[j] = neigh.z[j]

            self.b = self.b[:bundle_index]
            for j in tasks_to_remove:
                self.p.remove(j)
            self.S = self.S[:bundle_index+1]
            self.j_outbid = j
            self.outbidder = neigh.id
            self.outbid_pos = bundle_index
            self.their_net_cvg[0]=False
            self.length = self.evaluate_path(self.p, "RPT")
            if self.length<self.makespan:
                self.updated_makespan = self.makespan
            else:
                self.updated_makespan = self.length


    def reset(self, j):
        """
        reset procedure
        :param j: task j
        :return:
        """
        self.y[j] = self.min_val
        self.z[j] = None
        bundle = self.b
        if j in bundle:
            bundle_index = bundle.index(j)
            tasks_to_remove = self.b[bundle_index:]
            index_to_remove = [t for (i, t) in enumerate(bundle) if t in tasks_to_remove]

            self.y[j] = self.min_val
            self.z[j] = None

            self.y = [self.y[i] if i not in index_to_remove else self.min_val for i in range(self.nt)]
            self.z = [self.z[i] if i not in index_to_remove else None for i in range(self.nt)]

            self.b = self.b[:bundle_index]
            for j in tasks_to_remove:
                self.p.remove(j)
            if self.length < self.makespan:
                self.updated_makespan = self.makespan
            else:
                self.updated_makespan = self.length
            self.S = self.S[:bundle_index+1]
            self.their_net_cvg[0]= False

    def leave(self):
        """
        leave procedure, does nothing
        :return:
        """
        pass



    def resolve_conflict(self, all_agents, iter=0, index = "else"):
        """
        Resolution conflict phase (consensus)
        :param all_agents: list of all agents in the problem
        :param iter: current interation (for printing purposes only)
        :return:
        """
        neigh_index = np.argwhere(self.G[self.id, :] == 1).flatten()
        neigh_cvg = [True for _ in range(self.D)]
        for k in neigh_index:
            neigh = all_agents[k]
            #self.converged = neigh.converged
            for j in range(self.nt):
                # agent k (sender) thinks zkj is k
                if neigh.z[j] == neigh.id:
                    if self.z[j] == self.id:
                        if neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id <self.id):
                            update = True
                            if update == True:
                                self.update(neigh, j)
                    elif self.z[j] == k:
                        update = True
                        if update == True:
                            self.update(neigh, j)
                    # (unassigned/none)
                    elif self.z[j] == None:
                        update = True
                        if update == True:
                            self.update(neigh, j)
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m] or neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id <self.id):
                            self.update(neigh, j)
                # agent k (sender) thinks zkj is i
                elif neigh.z[j] == self.id:
                    if self.z[j] == self.id:
                        self.leave()
                    elif self.z[j] == k:
                        self.reset(j)
                    elif self.z[j] == None:
                        self.leave()
                    else:
                        m = self.z[j]
                        if neigh.s[m] > self.s[m]:
                            self.reset(j)
                elif neigh.z[j] == None:
                    if self.z[j] == self.id:
                        self.leave()
                    elif self.z[j] == k:
                        self.update(neigh, j)
                    elif self.z[j] == None:
                        self.leave()
                    else:
                        m = self.z[j]
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                # agent k (sender) thinks zkj is m not in  {i,k (, none)}
                else:
                    m = neigh.z[j]
                    if self.z[j] == self.id:
                        if neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j] or (  neigh.s[m] > self.s[m] and (neigh.y[j] == self.y[j] and neigh.id <self.id)):
                            self.update(neigh, j)
                    elif self.z[j] == k:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                        else:
                            self.reset(j)
                    elif self.z[j] == m:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                    elif self.z[j] == None:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                    else:
                        n = self.z[j]
                        #if n not in [self.id, neigh.id, m, None]:
                        if neigh.s[m] > self.s[m] and neigh.s[n] > self.s[n]:
                            self.update(neigh, j)
                        elif neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j] or (neigh.s[m] > self.s[m] and (neigh.y[j] == self.y[j] and neigh.id <self.id)):
                            self.update(neigh, j)
                        elif neigh.s[n] > self.s[n] and self.s[m] > neigh.s[m]:
                            self.reset(j)
            # update stamp if some updates/resets/leaves were made with a neighbor
            self.updated_makespan = max(self.updated_makespan, neigh.updated_makespan)
            self.compute_s(neigh, iter)
            for i in range(1, self.D):
                neigh_cvg[i] = neigh_cvg[i] and neigh.their_net_cvg[i-1]
        self.makespan = self.updated_makespan
        self.their_net_cvg[0] = (self.z == self.z_before)
        for i in range(1, self.D):
            self.their_net_cvg[i] = neigh_cvg[i] and self.their_net_cvg[i - 1]

        self.converged = self.their_net_cvg[-1]

        if self.converged==True:
            self.cvg_counter += 1

        if index == "last":
            self.y_before = copy.deepcopy(self.y)
            self.z_before = copy.deepcopy(self.z)
            self.flag_won = (len(self.p) != self.len_p_before)
            self.len_p_before = len(self.p)



if __name__ == "__main__":
    """
    To have GCBBA allocation: method = "global", detector = "none" or "decentralized" in launch_agents
    To switch to CBBA allocation: method = "baseline", detector = "none" or "centralized" in launch_agents
    """
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

    # communication graph initialization
    raw_graph, G = create_graph(na, p=0.5, graph_type="random", seed=seed)
    D = nx.diameter(raw_graph)
    agents, tasks = task_agent_init(na=na, nt=nt, pos_lim=xlim, sp_lim=sp_lim, dur_lim=dur_lim, lamb_lim=[0.95, 0.95],
                                    clim=[1, 1])
    orch_cbba= Orchestrator_CBBA(G, D, tasks, agents, Lt, metric=metric)

    # allocation launching
    t0 = time.time()
    assig, tot_score, makespan = orch_cbba.launch_agents(method="global", detector = "decentralized")
    tf0 = np.round(1000 * (time.time() - t0))

    print("GCBBA-{} total score. = {}; max score = {}; time = {} ms; assignment = {}".format(metric,tot_score, makespan, tf0,
                                                                                              assig))
    draw_paths(tasks, agents, assig, tot_score, title="GCBBA-{}".format(metric))








