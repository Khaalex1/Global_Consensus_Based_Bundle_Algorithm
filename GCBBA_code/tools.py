"""
File of auxiliary functions
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import cycle

class Task:
    """
    Task class, defined by an id, a position (x,y), duration, and lambda (only for TDR)
    """
    def __init__(self, id, char):
        self.id = id
        self.pos = np.array([char[0], char[1]])
        self.duration = char[2]
        self.lamb = char[3]

def check_conflict(assig, nb_tasks):
    """
    Checks conflict in assignment AND covering of all tasks (cardinality-optimal property)
    :param assig:
    :param nb_tasks:
    :return:
    """
    union = []
    for i in range(len(assig)):
        union += assig[i]
    set_union = set(union)
    if len(set_union)<nb_tasks:
        return True
    return False

def verify_connection(G):
    """
    Verifying communication graph is connex via laplacian eigenvalues
    :param G: communication matrix
    :return: True if G is connected, else False
    """
    # laplacian
    L = np.diag(np.array(np.sum(G, axis=1)).flatten()) - G
    eval, evec = np.linalg.eig(L)
    eval = np.sort(eval)
    return eval[1] > 1e-6

def create_graph(n, p=0.5, graph_type = "random", seed= None):
    """
    Creating a communication graph
    :param n: number of nodes (agents)
    :param p: probability of connection creation
    :param graph_type: "linear" => line graph, "full" => fully connected graph, else => random graph
    :param seed: None or value of seed
    :return: raw networkx graph, corresponding communication matrix G
    """
    if graph_type == "linear":
        raw_graph = nx.path_graph(n)
    elif graph_type == "full":
        raw_graph = nx.complete_graph(n)
    else:
        raw_graph = nx.fast_gnp_random_graph(n, p, seed=seed, directed=False)
    G = nx.adjacency_matrix(raw_graph).todense()
    while not verify_connection(G):
        seed += 1
        raw_graph = nx.fast_gnp_random_graph(n, p, seed=seed, directed=False)
        G = nx.adjacency_matrix(raw_graph).todense()
    return raw_graph, np.asarray(G.astype('float'))


def display_graph(raw_graph):
    """
    Plotting communication graph
    :param raw_graph: networkx graph
    :return:
    """
    plt.figure()
    plt.clf()
    nx.draw_networkx(raw_graph, with_labels=True)
    plt.title("Communication graph")
    plt.show()


def task_agent_init(na = 10, nt =100, pos_lim = [-5, 5], sp_lim = [1, 5], dur_lim = [1,5], lamb_lim = [0.95, 0.95], clim = [1, 1]):
    """
    Create agents and tasks characteristics
    :param na: nb of agents
    :param nt: nb of tasks
    :param pos_lim: (x,y) limits
    :param sp_lim: agent speed limits
    :param dur_lim: task duration limits
    :param lamb_lim: lambda (TDR) limits
    :param clim: weights (TDR) limits (useless in this implementation)
    :return:
    """
    agents = [np.concatenate((np.random.uniform(pos_lim[0], pos_lim[1], 2), np.random.uniform(sp_lim[0], sp_lim[1], 1)))
              for _ in range(na)]
    tasks = [np.concatenate(
        (np.random.uniform(pos_lim[0], pos_lim[1], 2), np.random.uniform(dur_lim[0], dur_lim[1], 1), np.random.uniform(lamb_lim[0], lamb_lim[1], 1),
         np.random.uniform(clim[0], clim[1], 1)))
        for _ in range(nt)]
    return agents, tasks


def draw_paths(pos_t, pos_a, assignment, score, title="Solution"):
    """
    drawing paths of agents along tasks according to assignment
    :param pos_t: list of positions of tasks
    :param pos_a: list of positions of agents
    :param xlim: x coordinate limit
    :param ylim: y coordinate limit
    :param assignment: list of assignments (paths) of each agent
    :param score: minsum value
    :param title: string
    :return:
    """
    cycol = cycle('bgrcmk')
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot()
    for j in range(len(pos_t)):
        task_point = [pos_t[j][0], pos_t[j][1]]
        ax.scatter([task_point[0]], [task_point[1]], color='black', marker='x', s=100)
        ax.text(task_point[0] - 0.010, task_point[1] + 0.25, "Task {}".format(j))
    for i in range(len(pos_a)):
        path = assignment[i]
        agent_point = [pos_a[i][0], pos_a[i][1]]
        plt.text(agent_point[0] - 0.015, agent_point[1] + 0.25, "Agent {}".format(i))
        c = next(cycol)
        for j in range(len(path)):
            task_index = assignment[i][j]
            task_point = pos_t[task_index]
            x_val = [agent_point[0], task_point[0]]
            y_val = [agent_point[1], task_point[1]]
            plt.plot(x_val, y_val, 'o', linestyle="--", c=c)
            plt.plot(x_val, y_val, 'o', linestyle="--", c=c)
            agent_point = task_point
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.set_title(title + "\nMin-Sum = {}".format(score))
    plt.show()

