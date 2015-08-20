# -*- coding: utf-8 -*-
#
# usage: python simulatoin.py SPEC_TYPE.py > SPEC_TYPE.nex
#
import sys
from collections import defaultdict
import numpy

NEXUS=1

class DiffusionSimulator(object):
    def __init__(self, spec, widx=0, init_state=True):
        self.birth = spec['birth']
        self.wordcost = spec['wordcost']

        self.id2name = []
        self.name2id = {}
        self.powers = []
        for name in spec['nodes']:
            self.name2id[name] = len(self.id2name)
            self.id2name.append(name)
            self.powers.append(spec['powers'][name])
        self.L = len(self.id2name)
        distances = {}
        for a, b, d in spec['distances']:
            ida = self.name2id[a]
            idb = self.name2id[b]
            if ida not in distances:
                distances[ida] = {}
            if idb not in distances:
                distances[idb] = {}
            distances[ida][idb] = d
            distances[idb][ida] = d

        self.weight = numpy.zeros((self.L, self.L))
        for i in xrange(self.L):
            for j in xrange(self.L):
                if i == j:
                    self.weight[i,j] = spec['scale'] * spec['survival'] * spec['powers'][self.id2name[i]]
                else:
                    if j in distances[i]:
                        self.weight[i,j] = spec['scale'] * spec['powers'][self.id2name[j]] / (distances[i][j] ** 2)
                    else:
                        self.weight[i,j] = 0.0
        if init_state:
            self.init_state(widx=widx)

    def init_state(self, widx=0):
        # initial state: every node has the same word widx
        self.state = [[widx] for i in xrange(self.L)]
        self.voc = {}
        self.voc[widx] = self.L
        self.wcounter = widx
        self.birth_record = { widx: 0 }
        self.death_record = {}

    def simulate(self, _iter=100):
        for __iter in xrange(_iter):
            voc_old = self.voc.copy() # for analysis

            state2 = [[] for i in xrange(self.L)]
            for i in xrange(self.L):
                # descrement
                for w in self.state[i]:
                    self.voc[w] -= 1

                wscores = {}
                for j in xrange(self.L):
                    if self.weight[i,j] <= 0.0:
                        continue
                    L = len(self.state[j])
                    for w in self.state[j]:
                        if w not in wscores:
                            wscores[w] = 0.0
                        wscores[w] += self.weight[i,j] / L
                wscores['new'] = self.birth * self.powers[i]

                # exhaustive enumeration  2^N - 1
                scores = [[[], 0.0]]
                for p, wscore in wscores.iteritems():
                    scores2 = []
                    for mem, s in scores:
                        scores2.append([mem, s])
                        mem2 = mem[:]
                        mem2.append(p)
                        scores2.append([mem2, s + wscore])
                    scores = scores2
                scores.pop(0) # remove dummy
                for t in scores:
                    t[1] -= len(t[0]) * (self.wordcost ** len(t[0]))
                idx = rand_partition_exp([s for mem, s in scores])
                mem = scores[idx][0]
                id_mem = []
                for a in sorted(mem):
                    if a == 'new':
                        self.wcounter += 1
                        id_mem.append(self.wcounter)
                        self.voc[self.wcounter] = 1
                    else:
                        id_mem.append(a)
                        self.voc[a] += 1
                # if len(id_mem) >= 2:
                #     print "%s\t%s" % (self.id2name[i], "\t".join([str(t) for t in id_mem]))
                state2[i] = id_mem

            self.state = state2
            for a, f in self.voc.items():
                assert(f >= 0)
                if f == 0:
                    del self.voc[a]
            # for analysis
            # voc_old
            for wid in set(self.voc.keys()) - set(voc_old.keys()):
                self.birth_record[wid] = __iter + 1
            for wid in set(voc_old.keys()) - set(self.voc.keys()):
                self.death_record[wid] = __iter + 1

        return self.state


def rand_partition_exp(prob_list):
    m = max(prob_list)
    e_list = [numpy.exp(s - m) for s in prob_list]
    s = sum(e_list)
    r = numpy.random.uniform(0, s)
    for i in xrange(0, len(e_list)):
        r -= e_list[i]
        if r <= 0.0:
            return i
    return len(prob_list) - 1

def to_nexus(spec, state_list):
    nchar = 0
    str_list = ["" for x in xrange(len(spec["nodes"]))]
    for state in state_list:
        voc = {}
        for w_list in state:
            for w in w_list:
                if w not in voc:
                    voc[w] = len(w_list)
        nchar += len(voc)
        for i, w_list in enumerate(state):
            for w in voc:
                if w in w_list:
                    str_list[i] += "1"
                else:
                    str_list[i] += "0"

    rv = "#nexus\r\nBEGIN DATA;\r\nDIMENSIONS ntax=%d nchar=%d;\r\nFORMAT\r\n\tdatatype=standard\r\n\tsymbols=\"01\"\r\n\tmissing=?\r\n\tgap=-\r\n\tinterleave=NO\r\n;\r\nMATRIX\n\n" % (len(spec['nodes']), nchar)
    for i, name in enumerate(spec["nodes"]):
        rv += "%s\t%s\r" % (name, str_list[i])
    rv += ";\r\nEND;"
    return rv

def update_survival_stat(survival_stat, sim, _iter=100, do_include_survived=False):
    for wid, birth in sim.birth_record.iteritems():
        if wid in sim.death_record:
            survival = sim.death_record[wid] - birth
            survival_stat[survival] += 1
        elif do_include_survived:
            survival = _iter + 1 - birth
            survival_stat[survival] += 1            


if __name__ == "__main__":
    numpy.random.seed(20)

    spec_path = sys.argv[1]
    execfile(spec_path)

    state_list = []
    survival_stat = defaultdict(int)
    for i in xrange(spec["itemsize"]):
        if 'type' in spec and spec['type'] == 'EVO':
            wcounter = 0
            for j, _spec in enumerate(spec['specs']):
                if j == 0:
                    sim = DiffusionSimulator(_spec, widx=wcounter)
                else:
                    # _spec = spec['specs'][0]
                    # break

                    sim_old = sim
                    sim = DiffusionSimulator(_spec, init_state=False)
                    sim.wcounter = sim_old.wcounter
                    sim.voc = sim_old.voc
                    sim.birth_record = sim_old.birth_record
                    sim.death_record = sim_old.death_record
                    new2old = {}
                    for name, _id in sim_old.name2id.iteritems():
                        _id2 = sim.name2id[name]
                        new2old[_id2] = _id
                    sim.state = [None for i in xrange(sim.L)]
                    for k in xrange(sim.L):
                        if k in new2old:
                            sim.state[k] = sim_old.state[new2old[k]][:]
                    for _from, _to in _spec['clones']:
                        sim.state[sim.name2id[_to]] = sim.state[sim.name2id[_from]][:]
                        for vid in sim.state[sim.name2id[_to]]:
                            sim.voc[vid] += 1

                if i == 0:
                    sys.stderr.write("weight matrix\n")
                    sys.stderr.write(str(sim.weight) + "\n")
                state = sim.simulate(_iter=_spec['steps'])
        else:
            _spec = spec
            sim = DiffusionSimulator(_spec)
            if i == 0:
                sys.stderr.write("weight matrix\n")
                sys.stderr.write(str(sim.weight) + "\n")
            state = sim.simulate(_iter=1000)
            # wcounter = sim.wcounter + 1

        update_survival_stat(survival_stat, sim)
        sys.stderr.write(str(state) + "\n")
        state_list.append(state)

    if NEXUS:
        sys.stdout.write(to_nexus(_spec, state_list))

    if False:
        import matplotlib.pyplot as plt
        # sys.stderr.write(str(survival_stat) + "\n")
        fig, ax = plt.subplots()
        x = sorted(survival_stat.keys())
        y = [survival_stat[k] for k in x]
        ax.plot(x, y, 'k--')
        plt.show()
