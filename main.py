import random
import streamlit as st
import pandas as pd
import numpy as np


class Chromosome:
    def __init__(self, gene, fitness):
        self.gene = gene
        self.fitness = fitness


def get_child_gene(index, parent, gene, mutate):
    if mutate and random.uniform(0, 1) > .5:
        return gene[index]
    return parent[index]


class Population:
    def __init__(self, population_size, gene_size, target):
        self.chromosomes = []
        self.population_size = population_size
        self.target = target
        self.gene_size = gene_size
        self.top_population_fitness = []

    def generate_population(self):
        for x in range(self.population_size):
            gene = np.random.uniform(low=0.0, high=self.target, size=(self.gene_size,)).round(2)
            fitness = get_fitness(gene, self.target)
            chromosome = Chromosome(gene, fitness)
            self.chromosomes.append(chromosome)

    def breed_population(self, survival_rate, mutate, allow_mixes):
        self.chromosomes.sort(key=lambda c: c.fitness)
        non_survivors = self.chromosomes[survival_rate:]
        self.chromosomes = self.chromosomes[:survival_rate]
        if allow_mixes:
            for i in range(int(survival_rate/3)):
                if random.uniform(0, 1) > .5:
                    index = int(random.uniform(0, len(self.chromosomes)))
                    self.chromosomes[index] = non_survivors[index]

        generated_children = []
        self.top_population_fitness.append([c.fitness for c in self.chromosomes])

        while len(generated_children) < self.population_size - survival_rate:
            parent_1 = random.choice(self.chromosomes)
            parent_2 = random.choice(self.chromosomes)
            child = self.breed_parents(parent_1, parent_2, mutate)
            generated_children.append(child)

        self.chromosomes.extend(generated_children)

    def breed_parents(self, parent_1, parent_2, mutate):
        child_gene = []
        gene = np.random.uniform(low=0.0, high=self.target, size=(self.gene_size,)).round(2)
        for i in range(len(parent_1.gene)):
            flip = random.uniform(0, 1)
            if flip > .5:
                child_gene.append(get_child_gene(i, parent_1.gene, gene, mutate))
            else:
                child_gene.append(get_child_gene(i, parent_2.gene, gene, mutate))
        return Chromosome(child_gene, get_fitness(child_gene, self.target))


def get_fitness(gene, target):
    fitness = 0
    var = 1
    for g in gene:
        fitness += var * g
        var += 1
    diff = abs(target - fitness) / 1
    diff = round(diff, 2)
    return diff


def get_result(gene):
    res = ''
    var = 1
    sum = 0
    for g in gene:
        res += '{x}*{num}+'.format(x=var, num=g)
        sum += var * g
        var += 1

    return res.rstrip(res[-1]), sum


def start():
    st.header('Solving equation with genetic algorithm')
    st.subheader('Example: Target = 30, Generated Variables = 4   1a+2b+3c+4d=30')
    pop_size = int(st.number_input('Enter The Population Size'))
    max_iterations = int(st.number_input('Enter The Maximum Allowed Iterations'))
    chromosome_gene_size = int(st.number_input('Enter The Number Of Generated Variables For The Equation'))
    rate_of_survival = int(st.number_input('Enter The Number Of Survivals Per Population'))
    target_number = st.number_input('Enter The Target Number For The Equation')
    allow_mutation = st.checkbox('Allow Mutation')
    allow_mixes_in_top_chromosomes = st.checkbox('Allow Dilution Of Top Chromosomes')
    top_chromosome = None
    p = Population(pop_size, chromosome_gene_size, target_number)
    if st.button('Run'):
        p.generate_population()
        fitness_per_population = []
        generation = []
        total_number_of_iterations = 1
        for i in range(max_iterations + 1):
            total_number_of_iterations = i
            p.breed_population(rate_of_survival, allow_mutation, allow_mixes_in_top_chromosomes)
            fitness_per_population.append([c.fitness for c in p.chromosomes])
            generation.append('iteration {index}'.format(index=i))
            top_chromosome = p.chromosomes[0]
            if top_chromosome.fitness == 0:
                st.snow()
                break

        st.subheader('Fitness Over Generations')
        df = pd.DataFrame(
            fitness_per_population, generation)
        st.line_chart(df)

        st.subheader('Fitness For Top Chromosomes Over Generations')
        top_fitness = pd.DataFrame(
            p.top_population_fitness, generation)
        st.line_chart(top_fitness)

        expression, total = get_result(top_chromosome.gene)
        st.write('Result After {iter} Iterations'.format(iter=total_number_of_iterations))
        st.latex(r'''
            {expr} = {res}
            '''.format(expr=expression, res=total))


start()
