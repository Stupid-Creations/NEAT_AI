import random

class NEAT:
    def __init__(self,input: int,output: int,innov:list):
        self.Nodes = []
        self.genes = []
        self.input = input
        self.output = output
        self.pred = []
        self.fitness = None
        self.innov = []

        for i in range(input):
            self.Nodes.append(Node(0,self))
        for i in range(output):
            self.Nodes.append(Node(2,self))
            for j in range(input):
                self.genes.append(Genome(self.Nodes[j],self.Nodes[-1],self,innov))
            self.Nodes[-1].update_genes(self)
        self.innov = [a.innov for a in self.genes]


    def activate(self,input):
        for i in range(2):
            self.pred = []
            for i in range(len(self.Nodes)):
                if self.Nodes[i].mode == 0:
                    self.Nodes[i].activate(input[i])
                else:
                    self.Nodes[i].activate()
                    if self.Nodes[i].mode == 2:
                        self.pred.append(self.Nodes[i].pred)

    def add_node(self,innov):
        choose = random.choice(self.genes)
        self.Nodes.append(Node(1,self))
        self.genes.append(Genome(choose.input,self.Nodes[-1],self,innov))
        self.genes[-1].weight = choose.weight
        self.genes.append(Genome(self.Nodes[-1],choose.output,self,innov))
        self.genes[-1].weight = 1
        self.genes[self.genes.index(choose)].enabled = False
        self.Nodes[-1].update_genes(self)
        self.Nodes[self.Nodes.index(choose.output)].update_genes(self)
        self.innov = [a.innov for a in self.genes]

    def rand_weight(self):
        choose = random.choice(self.genes)
        seed = random.random()
        if seed > 0.10:
            self.genes[self.genes.index(choose)].weight = self.genes[self.genes.index(choose)].weight+(random.random()/10)*random.choice([-1,1])
        else:
            self.genes[self.genes.index(choose)].weight = random.random()*random.choice([-1,1])
        self.Nodes[self.Nodes.index(choose.output)].update_genes(self)

    def add_connection(self,innov):
        cout = random.choice(self.Nodes)
        a = [b for b in self.Nodes if b!= cout]
        for i in a:
            if i.mode == 0:
                a.remove(i)
        for i in a:
            for j in cout.Nodes:
                if i == j:
                    a.remove(i)
        cinp = trychoice(a)
        if cinp == None:
            return 0
        self.genes.append(Genome(cinp,cout,self,innov))
        self.Nodes[self.Nodes.index(cout)].update_genes(self)
        self.innov = [a.innov for a in self.genes]

    def mutate(self,innov:list,seeds = None,Repeat = False):
        seed = random.random()
        if Repeat:
            seed = seeds
            print('a')
        if seed < 0.10:
            self.add_node(innov)
        elif seed < 0.35:
            a = self.add_connection(innov)
            print('b')
            if a == 0:
                seed += 0.9-seed
                self.mutate(innov)
        elif seed <= 1:
            self.rand_weight()
        
    def crossover(self, b,innov):
        if self.fitness > b.fitness:
            fitter = self
            not_fitter = b
        elif self.fitness < b.fitness:
            fitter = b
            not_fitter = self
        elif self.fitness == b.fitness:
            not_fitter = self
            fitter = b

        new = NEAT(self.input,self.output,innov)
        new.genes = []
        new.Nodes = []

        common = [a for a in self.genes if a.innov in b.innov]
        not_common = [a for a in fitter.genes if a.innov not in not_fitter.innov]

        for a in common:
            a.weight = random.random()
            if a.enabled == False:
                seed = random.random()
                if seed <= 0.25:
                    a.enabled = True
            new.genes.append(a)
        for a in not_common:
            a.weight = random.random()
            if a.enabled == False:
                seed = random.random()
                if seed <= 0.25:
                    a.enabled = True
            new.genes.append(a)

        for i in new.genes:
            if i.input not in new.Nodes:
                new.Nodes.append(i.input)
            if i.output not in new.Nodes:
                new.Nodes.append(i.output)
        new.mutate()
        return new

    def set_fitness(self,fitness):
        self.fitness = fitness

    def calc_compatibility(self,a,b,coeffEXCESS,coeffDISJOINT,coeffDIFF_MATCHING_WEIGHTS,compat_threshold):
        longer = max([len(a.genes),len(b.genes)])
        if longer == len(a.genes):
            long = a
        else:
            long = b

        matches = [x for x in a.innov if x in b.innov]
        last_match = max(matches)

        excess = len([x for x in long.genes if x.innov > last_match])
        disjoint = len([x for x in long.genes if x.innov < last_match])
        average_weight_diff = sum([a.weight-b.weight for a,b in zip(a.genes,b.genes) if a.innov in matches])/len(matches)

        return (coeffDISJOINT*disjoint)/longer+(coeffEXCESS*excess)/longer+(coeffDIFF_MATCHING_WEIGHTS*average_weight_diff) <= compat_threshold

    def compute_population(self,population: list):
        fitnesses = [a.fitness for a in population]
        adjusted_fits = [a/len(population) for a in fitnesses]
        mean_adjusted_fit = sum(adjusted_fits)/len(adjusted_fits)
        pop_size_current = sum(fitnesses)/mean_adjusted_fit
        new_pop = []
        population.sort(key = lambda x: x.fitness)
        can_repro = [a for a in population if population.index(a) < (pop_size_current/2)]
        for i in range(pop_size_current):
            new_pop.append(trychoice(can_repro).crossover(trychoice(can_repro)))
        return new_pop

sigmoid = lambda x: 1/(1+pow(2.7128,-x))

def trychoice(list):
    if len(list) > 1:
        return random.choice(list)
    elif len(list) == 1:
        return list[0]
    elif len(list) == 0:
        return None
    
class Node:
    def __init__(self,mode: int,NEAT: NEAT):
        # mode = 0 means input, 1 means hidden, 2 means output
        self.mode = mode
        self.genes = []
        self.pred = None
        self.update_genes(NEAT)
        self.Nodes = []

    def update_genes(self,NEAT: NEAT):
        self.genes = []
        for i in NEAT.genes:
            if i.output == self:
                self.genes.append(i)
                self.Nodes.append(i.input)

    def activate(self,input = None):
        if self.mode != 0:
            self.pred = sigmoid(sum([a.input.pred*a.weight for a in self.genes if a.enabled and a.input.pred]))
        else:
            self.pred = input

class Genome:
    def __init__ (self,input: Node,output: Node,NEAT: NEAT,innov: list):
        self.input = input
        self.output = output
        self.NEAT = NEAT
        self.innov = None
        self.enabled = True
        self.weight = random.random()
        
        self.inpdex = self.NEAT.Nodes.index(input)
        self.outdex = self.NEAT.Nodes.index(output)

        self.check_innov(innov)

    def check_innov(self,innov):
        if [self.inpdex,self.outdex] not in innov:
            innov.append([self.inpdex,self.outdex])
        self.innov = innov.index([self.inpdex,self.outdex])
