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

        for i in range(input+1):
            if i != input:
                self.Nodes.append(Node(0,self))
            if i == input:
                # 3 is bias node
                self.Nodes.append(Node(3,self))

        for i in range(output):
            self.Nodes.append(Node(2,self))
            for j in range(input+1):
                self.genes.append(Genome(self.Nodes[j],self.Nodes[-1],self,innov))
            self.Nodes[-1].update_genes(self)
        self.innov = [a.innov for a in self.genes]


    def activate(self,input):
        for i in range(2):
            self.pred = []
            inp_count = 0
            for i in range(len(self.Nodes)):
                if self.Nodes[i].mode == 0: 
                    self.Nodes[i].activate(input[inp_count])
                    inp_count+=1
                else:
                    if self.Nodes[i].mode != 3:
                        self.Nodes[i].activate()
                        if self.Nodes[i].mode == 2:
                            self.pred.append(self.Nodes[i].pred)
                    else:
                        self.Nodes[i].pred = 1

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
        if seed < 0.30:
            self.add_node(innov)
        elif seed < 0.65:
            try:
                a = self.add_connection(innov)
            except:
                seed += 0.20-seed
                self.mutate(innov,seed,True)
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
        new.mutate(innov)
        return new

    def set_fitness(self,fitness):
        self.fitness = fitness

def calc_compatibility(a,b,coeffEXCESS,coeffDISJOINT,coeffDIFF_MATCHING_WEIGHTS,compat_threshold):
    longer = max([len(a.genes),len(b.genes)])
    if longer == len(a.genes):
        long = a
    else:
        long = b
    if longer < 20:
        longer = 1

    matches = [x for x in a.innov if x in b.innov]
    last_match = max(matches)

    excess = len([x for x in long.genes if x.innov > last_match])
    disjoint = len([x for x in long.genes if x.innov < last_match])
    average_weight_diff = sum([a.weight-b.weight for a,b in zip(a.genes,b.genes) if a.innov in matches])/len(matches)

    return (coeffDISJOINT*disjoint)/longer+(coeffEXCESS*excess)/longer+(coeffDIFF_MATCHING_WEIGHTS*average_weight_diff) <= compat_threshold

    # in adjusted fitness, since we will make species in the main code, we can just reference species size here instead of comparing 
    # with full population each time
def compute_species(species: list,innov):
    fitnesses = [a.fitness for a in species]
    adjusted_fits = [a/len(species) for a in fitnesses]
    mean_adjusted_fit = sum(adjusted_fits)/len(adjusted_fits)
    pop_size_current = round(sum(fitnesses)/mean_adjusted_fit)
    new_pop = []
    species.sort(key = lambda x: x.fitness)
    #kill the underperformng 90% of the population
    can_repro = [a for a in species if species.index(a) < round((pop_size_current*9)/10)]
    for i in range(pop_size_current):
        new_pop.append(trychoice(can_repro).crossover(trychoice(can_repro),innov))
    return new_pop

def speciate_pop(s_population,c_of_wd,c_of_excess,c_of_disjoint,threshold):
    new_pop = []
    for i in s_population:
        done = False
        if len(new_pop)>0:
            for j in new_pop:
                # any one member is enough
                if calc_compatibility(i,j[0],c_of_excess,c_of_disjoint,c_of_wd,threshold):
                    new_pop[new_pop.index(j)].append(i)
                    done = True
                    break
            if not done:
                # make new species if not in existing species
                new_pop.append([i])
        else:
            new_pop.append([i])
    return new_pop

def update_pop(s_pop,c_of_wd,c_of_excess,c_of_disjoint,threshold,one_piece_of_data,answer,innov):
    n_s_pop = []
    n_us_pop = []
    MSE = lambda x,y: sum([(a[0]-a[1])**2 for a in zip(x,y)])/len(answer)
    # get the fitnesses
    for i in s_pop:
        for j in i:
            j.activate(one_piece_of_data)
            # use a - because we are using a cost function so the better the network, the smaller the cost
            j.set_fitness(-(MSE(j.pred,answer)))
    #make populations
    for i in s_pop:
        for j in compute_species(i,innov):
            n_us_pop.append(j)

    n_s_pop = speciate_pop(n_us_pop,c_of_wd,c_of_excess,c_of_disjoint,threshold)
    return n_s_pop

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
