#from env.forex_env import ForexEnv

#make df from 'GBPUSDe.csv'

#initialize Forex environment
env = ForexEnv(df)
#env = DummyVecEnv([lambda: ForexEnv(df)])

#observation will be used to determine the action after passing through the neural network
observation = env.reset()#.squeeze()
#print('line 20: observation:', observation.shape)

#the choice between mates with identical fitness will be random
random.seed()

#the genomes will all be stored in the dataframe in the form of {keys, fitness, nodes, connects}
genomes = pd.DataFrame()
genomes_old = pd.DataFrame()

#each node in nodes will have [h_mark, enabled, weight, bias]
#where h_mark is the historical marker created when a new node is created
#and shared by all nodes decended from the same node
#this makes mating and structual comparison much easier
#enabled is a 1 for an enabled node, 0 for a disabled node
#weight and bias are used as is standard for neural nets

#each connect in connects will have [h_mark, enabled, weight, bias, in_node, out_node]
#where h_mark, enabled, weight, and bias are exactly the same as with nodes
#in_node and out_node are the nodes which are connected by this connect

#Flexible number of inputs
#Tested with with 4 and 10 inputs
#Should work for 30 inputs, or any other number
num_inputs = 10

#number of generations to run for if we don't find a solution sooner
generations = 24

#hist_marker is the cornerstone of NEAT. It increments with each new node or connection 
#Allows comparison accross genes for mating and species division, by giving each
#node and connection a unique marker
hist_marker = 0

#gene_key is a unique ID for each genome, used for mating, fitness testing, mutation, and management
gene_key = 0

#hyperparamiters for neuroevolution of nodes and connections
weight_change_freq = 0.8
weight_uniform_cf = 0.9

#hyperparamiters for compatibility coefficients
c1 = 1.0
c2 = 1.0
c3 = 0.4
delta = 3.0

#generations allowed without improving fitness, if this number is exceded the 
#species is removed from reproduction pool
fitness_no_improve_limit = 15

#the champion of each species with more than this many networks is copied unchanged next generation
champion_copy_minimum = 5

#75% chance that gene disabled if disabled in either parent
disabled_inheritance = 0.75

#25% chance of offspring from pure mutation without crossover
mutate_vs_mate = 0.25

#interspecies mating rate 0.001
intermate_rate = 0.001

#new node and link addition probability
new_node_prob = 0.03
new_link_prob = 0.05

#maximum for normalization of % amount to buy/sell in step action
#initialized to 0, will follow maximum from action function
#there should be a better way, but this works for now
max_amount = 0

def distance_function(excess_gene_total, disjoint_gene_total, weight_dif_average, greater_gene_count):
    """
    compatability distance function, calculated based on gene differences rather than topolgy
    distance_function(total number of excess genes,
    total number of disjoint genes,
    average weight difference,
    the total number of genes in the larger of the two genomes)
    uses the c1, c2, and c3 compatablity coefficient hyperparamiters to sum up
    the effective "distance" between two geneomes
    in the original NEAT they state the settings for these hyperparamiters and
    to set gene counts of less than 20 egual to 1, which is done automatically here
    """
    global c1
    global c2
    global c3
    if(greater_gene_count < 20):
        greater_gene_count = 1
    arg1 = (c1 * excess_gene_total) / greater_gene_count
    arg2 = (c2 * disjoint_gene_total) / greater_gene_count
    arg3 = c3 * weight_dif_average
    return arg1 + arg2 + arg3

def compatability(a_nodes = [], a_connects = [], b_nodes = [], b_connects = []):
    """
    check if two genomes belong to the same species
    compatability([nodes from first genome],
    [connections from first genome],
    [nodes from second genome],
    [connections from second genome])
    """
    global delta
    E = num_excess_genes(a_nodes, a_connects, b_nodes, b_connects)
    D = num_disjoint_genes(a_nodes, a_connects, b_nodes, b_connects)
    W = average_weight_dif(a_nodes, a_connects, b_nodes, b_connects)
    N = max_num_genes(a_nodes, a_connects, b_nodes, b_connects)
    distance = distance_function(E, D, W, N)
    return (distance < delta)

def relu(z):
    """
    standard ReLu transfer/activation function
    returns z or 0 if z < 0
    included for testing
    """
    if(z < 0):
        z = 0
    return z
         
    
def sigmoid(z):
    """
    modified sigmoidal transfer function suggested by the original NEAT
    """
    return 1 / (1 + math.e ^ (-4.9 * z))

#this was for the Cartpole OpenAI Gym problem, not Forex NEAT
#the get_action function results in a 4 centered float, we need an action of 0 or 1
#this could be improved, but until then this binary_activation gets the job done
def binary_activation(z):
    """
    the get_action() function results in a 4.0 centered float, we need an action of 0 or 1
    this could be improved, but until then this binary_activation gets the job done
    this issue is not addressed in the original NEAT
    """
    if(z[0] > 4):
        return [1, z[1]]
    else:
        return [0, z[1]]

def normal_activation(z):
    global max_amount
    if z > max_amount:
        max_amount = z
    if z < 0:
        z = 0
    z /= max_amount
    return z

def num_excess_genes(a_nodes = [], a_connects = [], b_nodes = [], b_connects = []):
    """
    excess genes are defined as the extra genes found in the middle sections of two
    compared genomes, used for calculating the distance between the two genomes
    first we find the highest matching gene, then we sum the non matching genes
    between the beginning of each genome and the highest matching gene
    num
    """
    num_e = 0
    highest_match = 0
    for a_node in a_nodes:
        for b_node in b_nodes:
            if (a_node[0] == b_node[0]):
                highest_match = max(highest_match, a_node[0])
    for a_connect in a_connects:
        for b_connect in b_connects:
            if (a_connect[0] == b_connect[0]):
                highest_match = max(highest_match, a_connect[0])
    for a_node in a_nodes:
        if(a_node[0] > highest_match):
            num_e += 1
    for b_node in b_nodes:    
        if(b_node[0] > highest_match):
            num_e += 1
    for a_connect in a_connects:
        if(a_connect[0] > highest_match):
            num_e += 1
    for b_connect in b_connects:
        if(b_connect[0] > highest_match):
            num_e += 1
    return num_e

def num_disjoint_genes(a_nodes = [], a_connects = [], b_nodes = [], b_connects = []):
    num_d = 0
    highest_match = 0
    for a_node in a_nodes:
        for b_node in b_nodes:
            if (a_node[0] == b_node[0]):
                highest_match = max(highest_match, a_node[0])
    for a_connect in a_connects:
        for b_connect in b_connects:
            if (a_connect[0] == b_connect[0]):
                highest_match = max(highest_match, a_connect[0])
    for a_node in a_nodes:
        excess = True
        for b_node in b_nodes:
            if (a_node[0] == b_node[0]):
                excess = False
        if((excess) and (a_node[0] < highest_match)):
            num_d += 1
    for a_connect in a_connects:
        excess = True
        for b_connect in b_connects:
            if (a_connect[0] == b_connect[0]):
                excess = False
        if((excess) and (a_node[0] < highest_match)):
            num_d += 1
    for b_node in b_nodes:
        excess = True
        for a_node in a_nodes:
            if (a_node[0] == b_node[0]):
                excess = False
        if((excess) and (b_node[0] < highest_match)):
            num_d += 1
    for b_connect in b_connects:
        excess = True
        for a_connect in a_connects:
            if (a_connect[0] == b_connect[0]):
                excess = False
        if((excess) and (b_node[0] < highest_match)):
            num_d += 1
    return num_d

def average_weight_dif(a_nodes = [], a_connects = [], b_nodes = [], b_connects = []):
    match_count = 0
    weight_dif_sum = 0
    bias_dif_sum = 0
    for a_node in a_nodes:
        for b_node in b_nodes:
            if(a_node[0] == b_node[0]):
                match_count += 1
                weight_dif_sum += b_node[2] - a_node[2]
                bias_dif_sum = b_node[3] - a_node[3]
    for a_connect in a_connects:
        for b_connect in b_connects:
            if(a_connect[0] == b_connect[0]):
                match_count += 1
                weight_dif_sum += b_connect[2] - a_connect[2]
                bias_dif_sum = b_connect[3] - a_connect[3]
    ave_dif = (weight_dif_sum + bias_dif_sum)/ (2 * match_count)
    return ave_dif

def max_num_genes(a_nodes = [], a_connects = [], b_nodes = [], b_connects = []):
    return max(len(a_nodes) + len(a_connects), len(b_nodes) + len(b_connects))

#creates a new connection between in_node and out_node, initializes the weight and bias
#defaults enabled to 1(True) grants it a unique hist_marker and increments the hist_marker
#then it returns the set as a python list
def new_connect(in_node, out_node, init = False, i_ = 0):
    weight = 1
    bias = 1
    enabled = 1
    global hist_marker
    h_mark = hist_marker
    if(init and (hist_marker < 2 + num_inputs * 3)):
        hist_marker += 1
    elif(init):
        h_mark = i_
    elif(not init):
        hist_marker += 1
    return [h_mark, enabled, weight, bias, in_node, out_node]

def new_node(init = False, i_ = 0):
    bias = 1
    weight = 1
    enabled = 1
    global hist_marker
    h_mark = hist_marker
    if(init and (hist_marker < 3 + num_inputs)):
        hist_marker += 1
    elif(init):
        h_mark = i_
    elif(not init):
        hist_marker += 1
    return [h_mark, enabled, weight, bias]

def new_genome(nodes = [], connects = []):
    nodes_ = nodes
    connects_ = connects
    fitness = 0
    global gene_key
    key = gene_key
    gene_key += 1
    return [key, fitness, nodes_, connects_]
    
def add_connect(nodes = [], connects = []):
    num_ns = len(nodes)
    num_cs = len(connects)
    new_c = False
    while(not new_c):
        a = random.randint(0, num_ns)
        b = random.randint(2 + num_inputs, num_ns)
        if(b == 2 + num_inputs):
            b = 0
        while(a == b):
            b = random.randrange(num_ns)
            if(b == 5):
                b = 0
        new_c = True
        for row in connects:
            if((row[4] == a) and (row[5] == b)):
                new_c = False
    connects.append(new_connect(a, b))
    return [nodes, connects]

def add_node(nodes = [], connects = []):
    num_cs = len(connects)
    h, e, w, b = new_node()
    nodes.append([h, e, w, b])
    cs_pick = random.randint(0, num_cs)
    i = 0
    for row in connects:
        if(cs_pick == i):
            row[1] = 0
            connects.append(new_connect(row[4], h))
            c_h, c_e, c_w, c_b, c_i, c_o = new_connect(h, row[5])
            connects.append([c_h, c_e, row[3], row[4], c_i, c_o])
        i += 1
    return [nodes, connects]

def mutate_weights(nodes = [], connects = []):#and biases
    perturbation = random.uniform(-0.01, 0.01)
    for node in nodes:
        if(random.random() < weight_change_freq):
            if(random.random() < weight_uniform_cf):
                node[2] += perturbation
            else:
                node[2] = random.random()
            if(random.random() < weight_uniform_cf):
                node[3] += perturbation
            else:
                node[3] = random.random()
    for connect in connects:
        if(random.random() < weight_change_freq):
            if(random.random() < weight_uniform_cf):
                connect[2] += perturbation
            else:
                connect[2] = random.random()
            if(random.random() < weight_uniform_cf):
                connect[3] += perturbation
            else:
                connect[3] = random.random()
    return [nodes, connects]
                
def get_action(act_n, act_c, obs):
    action = old_get_action(0, act_n, act_c, obs)
    #amount = normal_activation(old_get_action(1, act_n, act_c, obs))
    amount = old_get_action(1, act_n, act_c, obs)
    return [action, amount]

def old_get_action(current_hist_marker, act_n, act_c, obs):
    is_node = False
    for row in act_n:
        if(row[0] == current_hist_marker):
            is_node = True
            if(current_hist_marker < num_inputs + 2) and (current_hist_marker > 1):
                hold = run_node(act_n[current_hist_marker][:], obs[current_hist_marker - 2])
                return hold
    out_ = 0
    for c_row in act_c:
        if((c_row[5] == current_hist_marker) and (c_row[1] == 1)):
            new_c = []
            for n_row in act_c:
                if(c_row != n_row):
                    new_c.append(n_row)
            hold = old_get_action(c_row[4], act_n, new_c, obs)
            out_ = run_node(c_row, hold)
    return out_

def run_node(ns, y):
    if(ns[1] == 1):
        return relu(y * ns[2] + ns[3])
    else:
        return y

#todo: make the inheritence of enable with chance to enable/disable nodes/connections
#todo: keep a list of inovations in the current generation to ensure that all identical inovations recieve the same historical marker
def mate(a_nodes, a_connects, a_fitness, b_nodes, b_connects, b_fitness):
    global hist_marker
    nodes = []
    connects = []
    for i in range(hist_marker + 1):
        in_a = False
        in_b = False
        for a_node in a_nodes:
            if(a_node[0] == i):
                in_a = True
                for b_node in b_nodes:
                    if(b_node[0] == i):
                        in_b = True
                        if(a_fitness > b_fitness):
                            nodes.append(a_node)
                        elif(a_fitness < b_fitness):
                            nodes.append(b_node)
                        else:
                            if(random.random() < 0.5):
                                nodes.append(a_node)
                            else:
                                nodes.append(b_node)
        if((not in_a) and (not in_b)):
            for b_node in b_nodes:
                if(b_node[0] == i):
                    in_b = True
                    nodes.append(b_node)
        for a_connect in a_connects:
            if(a_connect[0] == i):
                in_a = True
                for b_connect in b_connects:
                    if(b_connect[0] == i):
                        in_b = True
                        if(a_fitness > b_fitness):
                            connects.append(a_connect)
                        elif(a_fitness < b_fitness):
                            connects.append(b_connect)
                        else:
                            if(random.random() < 0.5):
                                connects.append(a_connect)
                            else:
                                connects.append(b_connect)
        if((not in_a) and (not in_b)):
            for b_connect in b_connects:
                if(b_connect[0] == i):
                    in_b = True
                    connects.append(b_connect)
        k, f, n, c = new_genome(nodes, connects)
    return [k, f, n, c]

#first generation initialization
for _ in range(150):
    nodes = []
    connects = []
    nodes.append(new_node(True, 0))
    nodes.append(new_node(True, 1))
    for i in range(2, 2 + num_inputs):
        nodes.append(new_node(True, i))
    for i in range(2 + num_inputs, 2 + 2 * num_inputs):
        connects.append(new_connect(i - num_inputs, 0, True, i))
    for i in range(2 * num_inputs + 2, num_inputs * 3 + 2):
        connects.append(new_connect(i - num_inputs * 2, 1, True, i))
    k, f, n, c = new_genome(nodes, connects)
    genomes = genomes.append({"keys": k,
                              "fitness": f,
                              "nodes": n,
                              "connects": c}, ignore_index=True)

#speciation initialization
species_count = 1
species_rep = []
species_rep.append(random.randrange(150))
species_list = []
for _ in range(150):
    species_list.append(0)
species_best = []
species_best.append(0)
species_improved = []
species_improved.append(0) 
species_total = []
species_total.append(150)
gens_since_improvement = []
gens_since_improvement.append(0)
best_ever = 0
min_fit = []
min_fit.append(500)

mating = False

for g in range(generations):
    print("Working Generation: ", g)
    genomes_old = genomes.copy()
    best = 0
    for i in range(species_count):
        species_improved[i] = 0
        species_best[i] = 0
        species_total[i] = 0
        for j in range(150):
            if(species_list[j] == i):
                if(species_total[i] == 0):
                    species_rep[i] = j#should be a random member, not the first, but this does not have a large effect
                    species_total[i] += 1
    for k in range(150):
        reward = 0
        r_total = 0
        completes = 0
        best = 0
        done = False
        steps = 0
        while (not done):
            steps += 1
            if(steps == 999) and (k == 149):
                render_buy = True
            action = get_action(genomes.loc[k, "nodes"], genomes.loc[k, "connects"], observation)
            #print('observation:', observation)
            #print('action:', action)
            observation, reward, done, _ = env.step(action)
            #observation = observation.squeeze()
            if(steps == 999) or (k == 149):
                print("Nodes: ", genomes.loc[k, "nodes"], "\nConnects: ", genomes.loc[k, "connects"])
                print('action:', action)
                print('reward:', reward / steps)
                env.render()
        observation = env.reset()#.squeeze()
        if g > 1:
            genomes.loc[k, "fitness"] = reward
            if(reward > species_best[species_list[k]]):
                species_best[species_list[k]] = reward
                gens_since_improvement[species_list[k]] = 0
                species_improved[species_list[k]] = 1
            if(reward > best):
                best = reward
                if(best > best_ever):
                    best_ever = best
                    print('*' * 60)
                    print("Best Yet Gen: ", g," Organism: ", k, " Gene Key: ", genomes.loc[k, "keys"], " Fitness: ", best_ever, "\nNodes: ", genomes.loc[k, "nodes"], "\nConnects: ", genomes.loc[k, "connects"])
                    print('*' * 60)
        
    #remove the lowest performing members of each species
    for i in range(species_count):
        min_fit[i] = 500
        gens_since_improvement[i] += not(species_improved[i])
    for k in range(150):
        if(genomes.loc[k, "fitness"] < min_fit[species_list[k]]):
            min_fit[species_list[k]] = genomes.loc[k, "fitness"]
    for k in range(150):
        if(genomes.loc[k, "fitness"] == min_fit[species_list[k]]):
            genomes.loc[k] = genomes.loc[species_rep[species_list[k]]]
    for k in range(150):
        chosen_mate = random.randint(0, 149)
        while(chosen_mate == k):
            chosen_mate = random.randint(0, 149)            
        if(gens_since_improvement[species_list[k]] < fitness_no_improve_limit):
            if((genomes_old.loc[k, "fitness"] < species_best[species_list[k]]) or (species_total[species_list[k]] <= 5)):
                n_, c_ = mutate_weights(genomes.loc[k, "nodes"], genomes.loc[k, "connects"])
                genomes.at[k, "nodes"] = n_
                genomes.at[k, "connects"] = c_
                if(random.random() < 0.75):
                    if(random.random() > 1 - new_node_prob):
                        genomes.at[k, "nodes"], genomes.at[k, "connects"] = add_node(genomes.loc[k, "nodes"], genomes.loc[k, "connects"])
                    if(random.random() > 1 - new_link_prob):
                        genomes.at[k, "nodes"], genomes.at[k, "connects"] = add_connect(genomes.loc[k, "nodes"], genomes.loc[k, "connects"])                        
                    if(random.random() < 0.999):
                        while((species_list[k] != species_list[chosen_mate]) or (chosen_mate == k)):
                            chosen_mate = random.randint(0, 149)
                    genomes.at[k, "keys"], genomes.at[k, "fitness"], genomes.at[k, "nodes"], genomes.at[k, "connects"] = mate(genomes.loc[k, "nodes"], genomes.loc[k, "connects"], genomes.loc[k, "fitness"], genomes_old.loc[chosen_mate, "nodes"], genomes_old.loc[chosen_mate, "connects"], genomes_old.loc[chosen_mate, "fitness"])
                    if(not mating):
                        print("Started mating. . .")
                        mating = True
                    compatability_check = False
                    for i in range(species_count):
                        if(compatability(genomes.loc[k, "nodes"], genomes.loc[k, "connects"], genomes.loc[species_rep[species_list[k]], "nodes"], genomes.loc[species_rep[species_list[k]], "connects"])):
                            compatability_check = True
                        if(not(compatability_check)):
                            species_count += 1
                            gens_since_1
                            improvement.append(0)
                            species_best.append(0)
                            species_improved.append(0)
                            species_rep[species_count - 1] = k
                            min_fit.append(500)

