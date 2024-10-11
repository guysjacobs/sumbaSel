f##Plot sel up down

import numpy as np
import pylab as pl
import matplotlib.ticker as mticker
import seaborn as sns
import copy
import pandas as pd
import statsmodels.api as sm


def hwe(p, q):
    return np.array([p*p, 2*p*q, q*q], dtype = float)

def calc_s_SEAOval(AA = 0.85, AB = 0.15, BB = 0.0, approach = 'afterbirth', runsel = False):
    assert (AA + AB + BB == 1.0)
    if approach == 'beforebirth':
        print("Appropriate is observation is after fertilisation but before birth")
        #p is determined based on AB freq and assumption of HW.
        p = (-1.0+(np.sqrt((1.0 - (4.0 * -1.0 * (-AB/2.0))))))/(-2.0)
        q = 1.0 - p
    if approach == 'afterbirth':
        print("Appropriate if observation is immediately after birth")
        assert BB == 0.0
        #Working through het = 2p(1-p)/[p^2 + 2p(1-p)] gives p = [2/(het - 2)] + 2
        #That het equation is simply HWE but getting rid of the q**2 term
        q = (2.0 / (AB - 2.0)) + 2.0
        p = 1.0 - q
    elif approach == 'aftersel':
        print("Appropriate if observation is immediately after birth and malarial selection")
        assert BB == 0.0
        #This is after selection, so we can just use the actual observed frequencies.
        #Really, we could use the actual observed frequencies in other cases too, eg might be appropriate for beforebirth especially
        p = (BB + BB + AB)/2.0
        q = 1.0 - p
    #Based on p = s1/(s1 + s2) and q = s2/(s1 + s2) in the Dobzhansky chapter of Harrison and Boyce 1972
    s1 = (1.0 - q)/float(q)
    if runsel == True:
        q_sim = copy.deepcopy(q)
        p_sim = copy.deepcopy(p)
        for gen in range(1,21):
            AA_fertilisation = q_sim**2
            AB_fertilisation = 2.0 * p_sim * q_sim
            BB_fertilisation = p_sim**2
            AA_birth = AA_fertilisation / float(AA_fertilisation + AB_fertilisation)
            AB_birth = AB_fertilisation / float(AA_fertilisation + AB_fertilisation)
            BB_birth = 0.0
            AA_postmal = (AA_birth * (1.0 - s1)) / ((AA_birth * (1.0 - s1)) + AB_birth)
            AB_postmal = AB_birth / ((AA_birth * (1.0 - s1)) + AB_birth)
            BB_postmal = 0.0
            p_sim = AB_postmal / 2.0
            q_sim = 1.0 - p_sim
            print("%d -- q = %.3f, p = %.3f; fertilisation: AA,AB,BB = %.3f,%.3f,%.3f; birth: AA,AB,BB = %.3f,%.3f,%.3f; malaria: AA,AB,BB = %.3f,%.3f,%.3f" %(gen, q_sim, p_sim, AA_fertilisation, AB_fertilisation, BB_fertilisation, AA_birth, AB_birth, BB_birth, AA_postmal, AB_postmal, BB_postmal))
    print("p", p, "q", q)
    print("s[1-x, 1, 0]; x = ", s1)
    print("s[1, 1+y, 0]; y = ", 1.0/(1.0-s1))
    return p, q, s1

def calculate_trace(init_freq = 0.01, s1_AA = 1.0, s1_AB = 1.0, s1_BB = 1.0, s2_AA = 1.0, s2_AB = 1.0, s2_BB = 1.0, t_total = 100, t_switch = 100, N = None, first_freq_targ = None):
    """
    Calculate a genotype and freq trace over t_total time steps assuming
    init_freq and HWE at start, discrete gens and infinite population size.
    At t_switch, change to s2 selection values.
    The process is to calculate the new genotype frequencies and then HWE (random mating) them.
    """
    geno_freq = hwe(1.0 - init_freq, init_freq)
    s1_array = np.array([s1_AA, s1_AB, s1_BB], dtype = float)
    s2_array = np.array([s2_AA, s2_AB, s2_BB], dtype = float)
    geno_trace = [copy.copy(geno_freq)]
    freq_trace = [copy.copy(init_freq)]
    
    t_phase1 = min(t_total, t_switch)
    for gen in range(t_phase1):
        geno_freq = geno_freq * s1_array
        geno_freq = geno_freq / np.sum(geno_freq)
        new_freq = np.array([geno_freq[0] + geno_freq[0] + geno_freq[1], geno_freq[1] + geno_freq[2] + geno_freq[2]]) / 2.0
        if type(N) != type(None):
            #Drift effects if we want.
            new_freq_A = np.random.binomial(n = N, p = new_freq[0]) / float(N)
            new_freq = np.array([new_freq_A, 1.0 - new_freq_A])
        
        geno_freq = hwe(new_freq[0], new_freq[1])
        freq_trace.append(new_freq[1])
        geno_trace.append(geno_freq)
        if type(first_freq_targ) != type(None):
            if freq_trace[-1] >= first_freq_targ:
                return gen
            elif freq_trace[-1] == 0:
                return np.nan
        
    t_phase2 = t_total - t_switch
    if t_phase2 > 0:
        for gen in range(t_phase2):
            geno_freq = geno_freq * s2_array
            geno_freq = geno_freq / np.sum(geno_freq)
            new_freq = np.array([geno_freq[0] + geno_freq[0] + geno_freq[1], geno_freq[1] + geno_freq[2] + geno_freq[2]]) / 2.0

            if type(N) != type(None):
                #Drift effects if we want.
                new_freq_A = np.random.binomial(n = N, p = new_freq[0]) / float(N)
                new_freq = np.array([new_freq_A, 1.0 - new_freq_A])
            
            geno_freq = hwe(new_freq[0], new_freq[1])
            freq_trace.append(new_freq[1])
            geno_trace.append(geno_freq)

    geno_trace = np.array(geno_trace)
    freq_trace = np.array(freq_trace)
    return freq_trace, [geno_trace]

def calculate_trace_migration(init_freq = 0.01, s1_AA = 1.0, s1_AB = 1.0, s1_BB = 1.0, s2_AA = 1.0, s2_AB = 1.0, s2_BB = 1.0, t_total = 100, t_switch = 100, N = None, m = 0, first_freq_targ = None):
    """
    Calculate a genotype and freq trace over t_total time steps assuming
    init_freq and HWE at start, discrete gens and infinite, or finite, population size.
    At t_switch, change to s2 selection values.
    The process is to calculate the new genotype frequencies and then HWE (random mating) them.
    This version returns two traces, with migration occuring between them
    """
    geno_freq_1 = hwe(1.0 - init_freq, init_freq)
    geno_freq_2 = hwe(1.0 - init_freq, init_freq)
    s1_array = np.array([s1_AA, s1_AB, s1_BB], dtype = float)
    s2_array = np.array([s2_AA, s2_AB, s2_BB], dtype = float)
    geno_trace_1 = [copy.copy(geno_freq_1)]
    freq_trace_1 = [copy.copy(init_freq)]
    geno_trace_2 = [copy.copy(geno_freq_2)]
    freq_trace_2 = [copy.copy(init_freq)]
    
    t_phase1 = min(t_total, t_switch)
    for gen in range(t_phase1):
        geno_freq_1 = geno_freq_1 * s1_array
        geno_freq_1 = geno_freq_1 / np.sum(geno_freq_1)
        new_freq_1 = np.array([geno_freq_1[0] + geno_freq_1[0] + geno_freq_1[1], geno_freq_1[1] + geno_freq_1[2] + geno_freq_1[2]]) / 2.0

        geno_freq_2 = geno_freq_2 * s1_array
        geno_freq_2 = geno_freq_2 / np.sum(geno_freq_2)
        new_freq_2 = np.array([geno_freq_2[0] + geno_freq_2[0] + geno_freq_2[1], geno_freq_2[1] + geno_freq_2[2] + geno_freq_2[2]]) / 2.0
        
        new_freq_1_mig = (new_freq_1 * (1.0 - m)) + (new_freq_2 * (m))
        new_freq_2_mig = (new_freq_2 * (1.0 - m)) + (new_freq_1 * (m))
        if type(N) != type(None):
            #Drift effects if we want.
            new_freq_1_mig_A = np.random.binomial(n = N, p = new_freq_1_mig[0]) / float(N)
            new_freq_1_mig = np.array([new_freq_1_mig_A, 1.0 - new_freq_1_mig_A])

            new_freq_2_mig_A = np.random.binomial(n = N, p = new_freq_2_mig[0]) / float(N)
            new_freq_2_mig = np.array([new_freq_2_mig_A, 1.0 - new_freq_2_mig_A])
        
        geno_freq_1 = hwe(new_freq_1_mig[0], new_freq_1_mig[1])
        freq_trace_1.append(new_freq_1_mig[1])
        geno_trace_1.append(geno_freq_1)

        geno_freq_2 = hwe(new_freq_2_mig[0], new_freq_2_mig[1])
        freq_trace_2.append(new_freq_2_mig[1])
        geno_trace_2.append(geno_freq_2)
        if type(first_freq_targ) != type(None):
            if freq_trace_1[-1] >= first_freq_targ or freq_trace_2[-1] >= first_freq_targ:
                return gen
            elif freq_trace_1[-1] == 0 or freq_trace_2[-1] == 0:
                return np.nan
        
    t_phase2 = t_total - t_switch
    if t_phase2 > 0:
        for gen in range(t_phase2):
            geno_freq_1 = geno_freq_1 * s2_array
            geno_freq_1 = geno_freq_1 / np.sum(geno_freq_1)
            new_freq_1 = np.array([geno_freq_1[0] + geno_freq_1[0] + geno_freq_1[1], geno_freq_1[1] + geno_freq_1[2] + geno_freq_1[2]]) / 2.0

            geno_freq_2 = geno_freq_2 * s2_array
            geno_freq_2 = geno_freq_2 / np.sum(geno_freq_2)
            new_freq_2 = np.array([geno_freq_2[0] + geno_freq_2[0] + geno_freq_2[1], geno_freq_2[1] + geno_freq_2[2] + geno_freq_2[2]]) / 2.0

            new_freq_1_mig = (new_freq_1 * (1.0 - m)) + (new_freq_2 * (m))
            new_freq_2_mig = (new_freq_2 * (1.0 - m)) + (new_freq_1 * (m))
        
            if type(N) != type(None):
                #Drift effects if we want.
                new_freq_1_mig_A = np.random.binomial(n = N, p = new_freq_1_mig[0]) / float(N)
                new_freq_1_mig = np.array([new_freq_1_mig_A, 1.0 - new_freq_1_mig_A])

                new_freq_2_mig_A = np.random.binomial(n = N, p = new_freq_2_mig[0]) / float(N)
                new_freq_2_mig = np.array([new_freq_2_mig_A, 1.0 - new_freq_2_mig_A])
            
            geno_freq_1 = hwe(new_freq_1_mig[0], new_freq_1_mig[1])
            freq_trace_1.append(new_freq_1_mig[1])
            geno_trace_1.append(geno_freq_1)

            geno_freq_2 = hwe(new_freq_2_mig[0], new_freq_2_mig[1])
            freq_trace_2.append(new_freq_2_mig[1])
            geno_trace_2.append(geno_freq_2)

    geno_trace_1 = np.array(geno_trace_1)
    freq_trace_1 = np.array(freq_trace_1)
    geno_trace_2 = np.array(geno_trace_2)
    freq_trace_2 = np.array(freq_trace_2)
    return [freq_trace_1, freq_trace_2], [geno_trace_1, geno_trace_2]

def calculate_trace_migration_N(init_freq = 0.01, s1_AA = 1.0, s1_AB = 1.0, s1_BB = 1.0, s2_AA = 1.0, s2_AB = 1.0, s2_BB = 1.0, t_total = 100, t_switch = 100, N = None, m = 0, pops = 2):
    """
    Calculate a genotype and freq trace over t_total time steps assuming
    init_freq and HWE at start, discrete gens and infinite, or finite, population size.
    At t_switch, change to s2 selection values.
    The process is to calculate the new genotype frequencies and then HWE (random mating) them.
    This version returns pop traces, with migration occuring between all populations.

    This implements binomial sampling to apply a population size.
    """
    assert pops >= 2
    geno_freqs = [hwe(1.0 - init_freq, init_freq) for i in range(pops)]
    s1_array = np.array([s1_AA, s1_AB, s1_BB], dtype = float)
    s2_array = np.array([s2_AA, s2_AB, s2_BB], dtype = float)
    geno_traces = [[copy.copy(geno_freqs[i]) for i in range(pops)]]
    freq_traces = [[copy.copy(init_freq) for i in range(pops)]]
    M = np.ones((pops, pops)) * m
    M[np.diag_indices_from(M)] = 1.0 - ((pops - 1) * m)
    #print(M)
    
    t_phase1 = min(t_total, t_switch)
    for gen in range(t_phase1):
        geno_freqs = [geno_freqs[i] * s1_array for i in range(pops)]
        geno_freqs = [geno_freqs[i] / np.sum(geno_freqs[i]) for i in range(pops)]
        new_freqs = np.array([np.array([geno_freqs[i][0] + geno_freqs[i][0] + geno_freqs[i][1], geno_freqs[i][1] + geno_freqs[i][2] + geno_freqs[i][2]]) / 2.0 for i in range(pops)])
        new_freqs_A = np.sum(new_freqs[::,0] * M, 1)
        new_freqs = list(zip(new_freqs_A, 1.0 - new_freqs_A))
        
        if type(N) != type(None):
            #Drift effects if we want.
            new_freqs_A = np.array([np.random.binomial(n = N, p = new_freqs[i][0]) / float(N) for i in range(pops)])
            new_freqs = list(zip(new_freqs_A, 1.0 - new_freqs_A))
        
        geno_freqs = [hwe(new_freqs[i][0], new_freqs[i][1]) for i in range(pops)]
        freq_traces.append([new_freqs[i][1] for i in range(pops)])
        geno_traces.append(geno_freqs)
        
    t_phase2 = t_total - t_switch
    if t_phase2 > 0:
        for gen in range(t_phase2):
            geno_freqs = [geno_freqs[i] * s2_array for i in range(pops)]
            geno_freqs = [geno_freqs[i] / np.sum(geno_freqs[i]) for i in range(pops)]
            new_freqs = np.array([np.array([geno_freqs[i][0] + geno_freqs[i][0] + geno_freqs[i][1], geno_freqs[i][1] + geno_freqs[i][2] + geno_freqs[i][2]]) / 2.0 for i in range(pops)])

            new_freqs_A = np.sum(new_freqs[::,0] * M, 1)
            new_freqs = list(zip(new_freqs_A, 1.0 - new_freqs_A))
            
            if type(N) != type(None):
                #Drift effects if we want.
                new_freqs_A = np.array([np.random.binomial(n = N, p = new_freqs[i][0]) / float(N) for i in range(pops)])
                new_freqs = list(zip(new_freqs_A, 1.0 - new_freqs_A))
            
            geno_freqs = [hwe(new_freqs[i][0], new_freqs[i][1]) for i in range(pops)]
            freq_traces.append([new_freqs[i][1] for i in range(pops)])
            geno_traces.append(geno_freqs)
    geno_traces = np.array(geno_traces)
    freq_traces = np.array(freq_traces)
    
    return freq_traces, [geno_traces]

def calculate_trace_X(init_freq = 0.01, s1_AA = 1.0, s1_AB = 1.0, s1_BB = 1.0, s2_AA = 1.0, s2_AB = 1.0, s2_BB = 1.0, t_total = 100, t_switch = 100, N = None, first_freq_targ = None):
    """
    G6PD has the AA ad
    Calculate a genotype and freq trace over t_total time steps assuming
    init_freq and HWE at start, discrete gens and infinite population size.
    At t_switch, change to s2 selection values.
    The process is to calculate the new genotype frequencies and then HWE (random mating) them.
    """
    
    male_female_proportions = np.array([0.5,0.5])
    male_geno = (1.0 - init_freq, init_freq)
    female_geno = hwe(1.0 - init_freq, init_freq)
    
    male_geno_normed = (1.0 - init_freq, init_freq)
    female_geno_normed = hwe(1.0 - init_freq, init_freq)
    
    s1_array_male = np.array([s1_AA, s1_BB], dtype = float)
    s2_array_male = np.array([s2_AA, s2_BB], dtype = float)
    s1_array_female = np.array([s1_AA, s1_AB, s1_BB], dtype = float)
    s2_array_female = np.array([s2_AA, s2_AB, s2_BB], dtype = float)

    male_geno_trace = [copy.copy(male_geno)]
    female_geno_trace = [copy.copy(female_geno)]

    alleles_count_A = (1.0 * male_female_proportions[0] * male_geno_normed[0]) + (2.0 * male_female_proportions[1] * female_geno_normed[0]) + (1.0 * male_female_proportions[1] * female_geno_normed[1])
    alleles_count_B = (1.0 * male_female_proportions[0] * male_geno_normed[1]) + (2.0 * male_female_proportions[1] * female_geno_normed[2]) + (1.0 * male_female_proportions[1] * female_geno_normed[1])

    freq_trace = [alleles_count_B / float(alleles_count_A + alleles_count_B)]
    
    t_phase1 = min(t_total, t_switch)
    for gen in range(t_phase1):
        #The order will be reproduction (taking into account drift if applied), then selection, then count
        
        #Reproduction. Note that we actually don't care about male and female freqs for the reproduction, every individual needs a male and female parent
        p_fXA = female_geno_normed[0] + (0.5 * female_geno_normed[1])
        p_fXB = 1.0 - p_fXA
        p_mXA = male_geno_normed[0]
        p_mXB = 1.0 - p_mXA

        #Finite population, if applied. The experienced frequencies of male and female drift according to respective N.
        if type(N) != type(None):
            #Drift effects if we want.
            p_mXA_new = np.random.binomial(n = N * (1.0 / 4.0), p = p_mXA) / float(N * (1.0/4.0))
            p_mXA, p_mXB = [p_mXA_new, 1.0 - p_mXA_new]

            p_fXA_new = np.random.binomial(n = N * (3.0 / 4.0), p = p_fXA) / float(N * (3.0/4.0))
            p_fXA, p_fXB = [p_fXA_new, 1.0 - p_fXA_new]
        

        #The only X is from the mother for males
        male_geno = np.array([p_fXA, p_fXB])
        #The females have two Xs
        female_geno = np.array([p_fXA * p_mXA, (p_fXA * p_mXB) + (p_fXB * p_mXA), p_fXB * p_mXB])

        #Selection
        male_geno = male_geno * s1_array_male
        female_geno = female_geno * s1_array_female
        male_geno_sum = float(np.sum(male_geno))
        female_geno_sum = float(np.sum(female_geno))

        male_female_proportions = np.array([male_geno_sum / (male_geno_sum + female_geno_sum), female_geno_sum / (male_geno_sum + female_geno_sum)])
        male_geno_normed = male_geno / male_geno_sum
        female_geno_normed = female_geno / female_geno_sum

                
        #Counting
        alleles_count_A = (1.0 * male_female_proportions[0] * male_geno_normed[0]) + (2.0 * male_female_proportions[1] * female_geno_normed[0]) + (1.0 * male_female_proportions[1] * female_geno_normed[1])
        alleles_count_B = (1.0 * male_female_proportions[0] * male_geno_normed[1]) + (2.0 * male_female_proportions[1] * female_geno_normed[2]) + (1.0 * male_female_proportions[1] * female_geno_normed[1])

        new_freq = np.array([alleles_count_A / float(alleles_count_A + alleles_count_B), alleles_count_B / float(alleles_count_A + alleles_count_B)])


        male_geno_trace.append(male_geno_normed)
        female_geno_trace.append(female_geno_normed)
        
        freq_trace.append(new_freq[1])

        if type(first_freq_targ) != type(None):
            if freq_trace[-1] >= first_freq_targ:
                return gen
            elif freq_trace[-1] == 0:
                return np.nan

    t_phase2 = t_total - t_switch
    if t_phase2 > 0:
        for gen in range(t_phase2):
            #The order will be reproduction (taking into account drift if applied), then selection, then count
            #Reproduction. Note that we actually don't care about male and female freqs for the reproduction, every individual needs a male and female parent
            p_fXA = female_geno_normed[0] + (0.5 * female_geno_normed[1])
            p_fXB = 1.0 - p_fXA
            p_mXA = male_geno_normed[0]
            p_mXB = 1.0 - p_mXA

            #Finite population, if applied. The experienced frequencies of male and female drift according to respective N.
            if type(N) != type(None):
                #Drift effects if we want.
                p_mXA_new = np.random.binomial(n = N * (1.0 / 4.0), p = p_mXA) / float(N * (1.0/4.0))
                p_mXA, p_mXB = [p_mXA_new, 1.0 - p_mXA_new]

                p_fXA_new = np.random.binomial(n = N * (3.0 / 4.0), p = p_fXA) / float(N * (3.0/4.0))
                p_fXA, p_fXB = [p_fXA_new, 1.0 - p_fXA_new]
        
            #The only X is from the mother for males
            male_geno = np.array([p_fXA, p_fXB])
            #The females have two Xs
            female_geno = np.array([p_fXA * p_mXA, (p_fXA * p_mXB) + (p_fXB * p_mXA), p_fXB * p_mXB])

            #Selection
            male_geno = male_geno * s2_array_male
            female_geno = female_geno * s2_array_female
            male_geno_sum = float(np.sum(male_geno))
            female_geno_sum = float(np.sum(female_geno))

            male_female_proportions = np.array([male_geno_sum / (male_geno_sum + female_geno_sum), female_geno_sum / (male_geno_sum + female_geno_sum)])
            male_geno_normed = male_geno / male_geno_sum
            female_geno_normed = female_geno / female_geno_sum

            #Counting
            alleles_count_A = (1.0 * male_female_proportions[0] * male_geno_normed[0]) + (2.0 * male_female_proportions[1] * female_geno_normed[0]) + (1.0 * male_female_proportions[1] * female_geno_normed[1])
            alleles_count_B = (1.0 * male_female_proportions[0] * male_geno_normed[1]) + (2.0 * male_female_proportions[1] * female_geno_normed[2]) + (1.0 * male_female_proportions[1] * female_geno_normed[1])

            new_freq = np.array([alleles_count_A / float(alleles_count_A + alleles_count_B), alleles_count_B / float(alleles_count_A + alleles_count_B)])

            male_geno_trace.append(male_geno_normed)
            female_geno_trace.append(female_geno_normed)
            
            freq_trace.append(new_freq[1])

    male_geno_trace = np.array(male_geno_trace)
    female_geno_trace = np.array(female_geno_trace)
    freq_trace = np.array(freq_trace)
    return freq_trace, [female_geno_trace, male_geno_trace]


def plot_trace(axis = None, freq_trace = [], label = None, gentime = 1, plot_color = sns.color_palette()[0], alpha = 1.0, lw = 1):
    """
    Add a trace to the axis.
    """
    #print(list(range(len(freq_trace))), freq_trace)
    axis.plot(np.array(range(len(freq_trace))) * gentime, freq_trace, label = label, color = plot_color, alpha = alpha, lw = lw)
    return axis


def first_freq_dist(conditions = [], t_total = 100, selection_mode = 'diploid', gentime = 1, N = None, first_freq_reps = 100, first_freq_vals = []):
    """
    Conditions = list of [init_freq,t_introduced,s1AA,s1AB,s1BB,s2AA,s2AB,s2BB]

    Call traces based on selection on a diploid or X chromosome locus give different selection conditions and retrieve a distribution of first-hit times
    """
    first_freq_dists = []
    for condition_idx in range(len(conditions)):
        condition = conditions[condition_idx]
        first_freq_dists.append([])
        if selection_mode == 'diploid':
            while len(first_freq_dists[-1]) < first_freq_reps:
                first_freq = calculate_trace(init_freq = condition[0], s1_AA = condition[2], s1_AB = condition[3], s1_BB = condition[4], s2_AA = condition[5], s2_AB = condition[6], s2_BB = condition[7], t_total = t_total, t_switch = t_total, N = N, first_freq_targ = first_freq_vals[condition_idx])
                if np.isnan(first_freq) == False:
                    first_freq_dists[-1].append(first_freq)
            first_freq_dists[-1] = np.array(first_freq_dists[-1])
        elif selection_mode == 'xchrom':
            while len(first_freq_dists[-1]) < first_freq_reps:
                first_freq = calculate_trace_X(init_freq = condition[0], s1_AA = condition[2], s1_AB = condition[3], s1_BB = condition[4], s2_AA = condition[5], s2_AB = condition[6], s2_BB = condition[7], t_total = t_total, t_switch = t_total, N = N, first_freq_targ = first_freq_vals[condition_idx])
                if np.isnan(first_freq) == False:
                    first_freq_dists[-1].append(first_freq)
            first_freq_dists[-1] = np.array(first_freq_dists[-1])
        elif selection_mode == 'alphathal':
            while len(first_freq_dists[-1]) < first_freq_reps:
                first_freq = freq_geno_trace_finite = calculate_trace_alphaThal(init_freq_d1 = condition[0], init_freq_d2 = condition[1], s1_d0 = condition[3], s1_d1 = condition[4], s1_d2 = condition[5], s1_d3 = condition[6], s1_d4 = condition[7], s2_d0 = condition[8], s2_d1 = condition[9], s2_d2 = condition[10], s2_d3 = condition[11], s2_d4 = condition[12], t_total = t_total, t_switch = t_total, N = N, first_freq_targ = first_freq_vals[condition_idx])
                if np.isnan(first_freq) == False:
                    first_freq_dists[-1].append(first_freq)
            first_freq_dists[-1] = np.array(first_freq_dists[-1])
    first_freq_dists = np.array(first_freq_dists)
    return first_freq_dists

def plot_figure_trace(conditions = [], t_total = 100, t_switch = 50, selection_mode = 'diploid', gentime = 1, h_lines = [], h_labels = [], show_legend = True, one_color = False, y_max = None, N = None, N_reps = 0, plot_deterministic = None, show_figure = True):
    """
    Conditions = list of [init_freq,t_introduced,s1AA,s1AB,s1BB,s2AA,s2AB,s2BB]
    
    Call traces based on selection on a diploid or X chromosome locus give different selection conditions and plot them.
    """
    freq_traces_infinite = []
    freq_traces_finite = []
    trace_labels = ['q0=%.4f,t_in=%d,s1_AA=%.3f,s1_AB=%.3f,s1_BB=%.3f,s2_AA=%.3f,s2_AB=%.3f,s2_BB=%.3f' %(conditions[i][0], conditions[i][1], conditions[i][2] - 1.0, conditions[i][3] - 1.0, conditions[i][4] - 1.0, conditions[i][5] - 1.0, conditions[i][6] - 1.0, conditions[i][7] - 1.0) for i in range(len(conditions))]
    
    for condition_idx in range(len(conditions)):
        condition = conditions[condition_idx]
        if selection_mode == 'diploid':
            #Get the deterministic traces
            freq_geno_trace_infinite = calculate_trace(init_freq = condition[0], s1_AA = condition[2], s1_AB = condition[3], s1_BB = condition[4], s2_AA = condition[5], s2_AB = condition[6], s2_BB = condition[7], t_total = t_total - condition[1], t_switch = t_switch - condition[1], N = None)
            freq_traces_infinite.append([np.nan for i in range(condition[1])] + list(freq_geno_trace_infinite[0]))
            if type(N) != type(None):
                #Do stochastic traces too
                for run in range(N_reps):
                    freq_geno_trace_finite = calculate_trace(init_freq = condition[0], s1_AA = condition[2], s1_AB = condition[3], s1_BB = condition[4], s2_AA = condition[5], s2_AB = condition[6], s2_BB = condition[7], t_total = t_total - condition[1], t_switch = t_switch - condition[1], N = N)
                    freq_traces_finite.append(np.array([np.nan for i in range(condition[1])] + list(freq_geno_trace_finite[0])))
                        
                    
        elif selection_mode == 'xchrom':
            freq_geno_trace_infinite = calculate_trace_X(init_freq = condition[0], s1_AA = condition[2], s1_AB = condition[3], s1_BB = condition[4], s2_AA = condition[5], s2_AB = condition[6], s2_BB = condition[7], t_total = t_total - condition[1], t_switch = t_switch - condition[1])
            freq_traces_infinite.append(np.array([np.nan for i in range(condition[1])] + list(freq_geno_trace_infinite[0])))
            if type(N) != type(None):
                #Do stochastic traces too
                for run in range(N_reps):
                    freq_geno_trace_finite = calculate_trace_X(init_freq = condition[0], s1_AA = condition[2], s1_AB = condition[3], s1_BB = condition[4], s2_AA = condition[5], s2_AB = condition[6], s2_BB = condition[7], t_total = t_total - condition[1], t_switch = t_switch - condition[1], N = N)
                    freq_traces_finite.append(np.array([np.nan for i in range(condition[1])] + list(freq_geno_trace_finite[0])))
                    

    fig, ax = pl.subplots()
    if y_max == None:
        if type(N) != type(None):
            y_max = min(np.nanmax(freq_traces_finite) * 1.2, 1.0)
        else:
            y_max = min(np.nanmax(freq_traces_infinite) * 1.2, 1.0)
    x_max = t_total * gentime
    x_ticks = np.linspace(0, x_max, 5)
    ax.set_xticks(x_ticks)
    
    for condition_idx in range(len(conditions)):
        if type(N) != type(None):
            for run in range(N_reps):
                plot_color = sns.color_palette()[condition_idx] if one_color == False else sns.color_palette()[0]
                plot_trace(ax, freq_traces_finite[(condition_idx * N_reps) + run], trace_labels[condition_idx], gentime = gentime, plot_color = plot_color, alpha = 1.0 / np.log2(N_reps + 1))
            if plot_deterministic == True:
                plot_color = sns.color_palette()[condition_idx] if one_color == False else sns.color_palette()[0]
                plot_trace(ax, freq_traces_infinite[condition_idx], trace_labels[condition_idx], gentime = gentime, plot_color = plot_color, alpha = 1.0, lw = 2)
        else:
            plot_color = sns.color_palette()[condition_idx] if one_color == False else sns.color_palette()[0]
            plot_trace(ax, freq_traces_infinite[condition_idx], trace_labels[condition_idx], gentime = gentime, plot_color = plot_color, alpha = 1.0)

    if t_switch < t_total:
        ax.vlines(t_switch * gentime, 0.0, 1.0, colors = 'DarkRed', linestyles = 'dashed')


    for h_idx in range(len(h_lines)):
        #This assumes we are talking about freq d1
        ax.hlines(h_lines[h_idx],0, t_total * gentime, colors = 'DarkBlue', linestyles = ':')
        ax.text(0, h_lines[h_idx] + (0.02 * y_max), s = h_labels[h_idx])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos : int(x - (t_switch * gentime))))

    ax.set_ylim([0.0,y_max])
    pl.ylabel('Frequency')
    pl.xlabel('Time (years)')
    if show_legend:    
        pl.legend()
    if show_figure:
        pl.show()
    else:
        pl.clf()
    return np.array(freq_traces_infinite), np.array(freq_traces_finite)



#### Sumba paper SAO
"""
Sample location | Heterozygotes | SAO allele frequency | Selection coefficient (after selection) | Sel coef (after sel), het adv
East | 24/210 | 5.7% | 0.061 | 0.0645
East | 9/120 | 3.75% | 0.039 | 0.041
East | 46/360 | 6.4% | 0.068 | 0.073
East | 8/83 | 4.8% | 0.051 | 0.053
Southwest | 15/73 | 10.3% | 0.115 | 0.129
Southwest | 167/971 | 8.6% | 0.094 | 0.104
Southwest | 78/610 | 6.4% | 0.068 | 0.073
Southwest | 4/19 | 10.5% | 0.117 | 0.133
"""


#Outlier analysis asking whether SAO frequency differences imply different selection regimes
"""
p_vals_dict = {}
Ne_v_to_test = list(np.arange(50,1501, 50)) + [179, 1658]
for Ne_v in  Ne_v_to_test:
    for m_v in [0.015, 0.055]:
        p_vals = []
        tot_sig = 0
        for i in range(1000):
            if i % 100 == 0:
                print(Ne_v, m_v, "rep: ", i)
            trace_given_migration = calculate_trace_migration_N(init_freq = 0.143 / 2.0,
                                  s1_AA = 1.0,
                                  s1_AB = 1.084,
                                  s1_BB = 0.0,
                                  s2_AA = 1.0,
                                  s2_AB = 1.0,
                                  s2_BB = 0.0,
                                  t_total = 162,
                                  t_switch = 163,
                                  N = Ne_v * 2, #N = int(np.round((1435*2)/8.0)),
                                  m = m_v,
                                  pops = 8)
            #print(trace_given_migration[0][0], trace_given_migration[0][161])
            freq_1 = np.average([trace_given_migration[0][161][0], trace_given_migration[0][161][1], trace_given_migration[0][161][2], trace_given_migration[0][161][3]])
            freq_2 = np.average([trace_given_migration[0][161][4], trace_given_migration[0][161][5], trace_given_migration[0][161][6], trace_given_migration[0][161][7]])
            #freq_1 = trace_given_migration[0][161][0]
            #freq_2 = trace_given_migration[0][161][1]
            
            east_het = np.array([np.random.binomial(210, freq_1), np.random.binomial(120, freq_1), np.random.binomial(360, freq_1), np.random.binomial(83, freq_1)])
            west_het = np.array([np.random.binomial(73, freq_2), np.random.binomial(971, freq_2), np.random.binomial(610, freq_2), np.random.binomial(19, freq_2)])
            east_hom = np.array([210,120,360,83]) - east_het
            west_hom = np.array([73,971,610,19]) - west_het
            het = np.concatenate((east_het, west_het), dtype = int)
            hom = np.concatenate((east_hom, west_hom), dtype = int)
            region = [0,0,0,0,1,1,1,1]
            data = pd.DataFrame(data = list(zip(het, hom, region)), columns = ["Het", "Hom", "Region"])
            data_endog = data[["Het", "Hom"]]
            data_exog = data[["Region"]]
            data_exog = sm.add_constant(data_exog, prepend=False)
            glm_binom = sm.GLM(data_endog, data_exog, family=sm.families.Binomial())
            res = glm_binom.fit()
            p_vals.append(res.pvalues["Region"])
            if res.pvalues["Region"] < 0.0031315359698687365:
                tot_sig += 1
        if m_v not in p_vals_dict.keys():
            p_vals_dict[m_v] = {}
        p_vals_dict[m_v][Ne_v] = tot_sig

p_vals_dict_df = pd.DataFrame.from_dict(p_vals_dict) / 1000

"""

#Frequency traces SAO
"""
fig_S1A = plot_figure_trace(conditions = [[0.01,0,1.0,1.064,0.0,1.0,1.0,0.0],
                   [0.01,0,1.0,1.094,0.0,1.0,1.0,0.0]
                   ],
     t_total = 200,
     t_switch = 100,
     gentime = 25,
     h_lines = [0.0565, 0.079],
     selection_mode = 'diploid',
     h_labels = ['East',
                 'Southwest'],
     show_legend = False,
     one_color = False, y_max = 0.12)


fig_S1B = plot_figure_trace(conditions = [
                   [0.01,0,1.0,1.094,0.0,1.0,1.0,0.0]
                   ],
     t_total = 200,
     t_switch = 100,
     gentime = 25,
     h_lines = [0.0565, 0.079],
     selection_mode = 'diploid',
     h_labels = ['East',
                 'Southwest'],
     show_legend = False,
     one_color = False,
     N = 1500*2,
     N_reps = 100,
     y_max = 0.12,
     show_figure = True)


a = plot_figure_trace(conditions = [
                   [0.01,0,1.0,1.094,0.0,1.0,1.0,0.0]
                   ],
     t_total = 800, #200
     t_switch = 400, #100
     gentime = 25,
     h_lines = [0.0565, 0.079],
     selection_mode = 'diploid',
     h_labels = ['East',
                 'Southwest'],
     show_legend = False,
     one_color = False,
     N = 1500*2,
     N_reps = 1000,
     y_max = 0.12,
     show_figure = False)


first_past_95 = []
first_extinction = []
for i in range(len(a[1])):
    try:
        first_past_95.append(np.where(a[1][i] > (0.158/2. * 0.95))[0][0] * 25)
        first_extinction.append(np.where(a[1][i][400:] == 0.0)[0][0] * 25)
    except:
        pass
    
print(len(first_past_95), np.average(first_past_95), np.percentile(first_past_95, 2.5), np.percentile(first_past_95, 97.5))
#Paper run:
#9940 1300.4577464788733 600.0 2450.0
print(len(first_extinction), np.average(first_extinction), np.percentile(first_extinction, 2.5), np.percentile(first_extinction, 97.5))
#Paper run:
#9845 3354.5962417470796 1225.0 7775.0
print(np.sort(first_extinction)[int(len(first_extinction) / 2)])
#Paper run:
#2975
"""


###G6PDd
#The most detailed estimation is from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5222559/ which attempts to pull apart cerebral and anaemia malaria
#This data is based on G6PD+202T
#There is some information on cerebral malaria vs severe malarial anaemia.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3391056/
#We apply a 50:50 weigthing.

#Fitness estimates for cerebral are:
#A1A1  A1A2 A2A2   A1(m)   A2(m)
#0.974  1   0.974  0.964    1
#Fitness estimates for anaemia are:
#A1A1  A1A2 A2A2   A1(m)   A2(m)
#1      1   0.954  1      0.968

#Assuming 50/50 rate of the two diseases:
#0.987  1   0.965  0.982  0.984

#Given the modelling appraoch and 50% of time in each sex, approximated using average A1A1(f),A1(m) and A2A2(f),A2(f)
#(0.987 + 0.982) / 2.0 = 0.9845
#(0.965 + 0.984) / 2.0 = 0.9745

"""
fig_S2 = plot_figure_trace(conditions = [[0.01,60,0.9845,1.0,0.9745,1.0,1.0,0.99]
                   ],
     t_total = 1000,
     t_switch = 500,
     gentime = 25,
     selection_mode = 'xchrom',
     h_lines = [0.105, 0.038],
     h_labels = ['Coastal',
                 'Inland'],
     show_legend = False,
     one_color = False,
     N = 1500*2,
     N_reps = 20,
     plot_deterministic = True
     )


first_freq_dist_G6PD_Inland = first_freq_dist(conditions = [[0.01,0,0.9845,1.0,0.974,1.0,1.0,0.99]],
               t_total = 10000,
               gentime = 25,
               selection_mode = 'xchrom',
               N = 1500*2,
               first_freq_reps = 1000,
               first_freq_vals = [0.105],
               )[0]

print("First freq distribution for G6PD allele to reach 0.105 in Inland Sumba: mean = %d, 2.5 = %d, 5 = %d, 25 = %d, 50 = %d, 75 = %d, 95 = %d, 97.5 = %d" %(np.mean(first_freq_dist_G6PD_Inland) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 2.5) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 5) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 25) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 50) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 75) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 95) * 25,
                                                                                                                                                                    np.percentile(first_freq_dist_G6PD_Inland, 97.5) * 25))

"""
