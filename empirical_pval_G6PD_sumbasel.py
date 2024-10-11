##Calculating empirical p value for the observed freq difference in G6PD using pre-calculated DAF files.

##GSJ v1.0 29th Jul 2022
##GSJ v1.1 8th April 2023 adding plotting functions

import numpy as np
import gzip
import pylab as pl
import seaborn as sns

DAF_AUTO_FOLDER = '/media/guy/sf_Genetic_Data/SumbaIlluminaData/freq/'
DAF_X_FOLDER = '/media/guy/sf_Genetic_Data/SumbaIlluminaData/freq/X_chromosome/'

NUM_SNPS = 100

print("Folder containing derived allele frequencies (auto) set to ", DAF_AUTO_FOLDER)
print("Folder containing derived allele frequencies (X) set to ", DAF_X_FOLDER)

print("Reading in autosome data")

freq_file = gzip.open(DAF_AUTO_FOLDER + 'INDONESIA_SUMBA_TIMOR.geno0.05.chr1_22.removeGlobalFixed.DAF.txt.gz', 'rb')
freq_file.readline().decode()[0:-2].split('\t')


#['CHR', 'POS', 'SNP', 'A1', 'A2', 'Sumba_Anakalang', 'Sumba_Kodi', 'Sumba_Lamboya', 'Sumba_Loli', 'Sumba_Mamboro', 'Sumba_Rindi', 'Sumba_Wanokaka', 'Sumba_Wunga']
freq_diffs = []
snp_list = []
freqs = []

#Read in the data. Freq is taken as the average over inland, or the average over coastal. Choice of populations should be confirmed.
for line in freq_file:
	split_line = line.decode()[0:-2].split('\t')
	av_A = np.average(np.array([split_line[6], split_line[7], split_line[9], split_line[10], split_line[11]], dtype = float)) #Sumba_Kodi, Sumba_Lamboya, Sumba_Mamboro, Sumba_Rindi, Sumba_Wanokaka
	av_B = np.average(np.array([split_line[5], split_line[8], split_line[12]], dtype = float)) #Sumba_Anakalang, Sumba_Loli, Sumba_Wunga
	snp_list.append([split_line[0:5]])
	freqs.append(np.array(split_line[5:], dtype = float))
	freq_diffs.append([av_A, av_B])

freq_diffs = np.array(freq_diffs)
snp_list = np.array(snp_list)
freqs = np.array(freqs)

print("Reading in X chromosome data")

###Read in the X chromosome
freq_file_x = gzip.open(DAF_X_FOLDER + 'INDONESIA_SUMBA_TIMOR.geno0.05.chr_23_noFemale.removeGlobalFixed.DAF.txt.gz', 'rb')
freq_file_x.readline().decode()[0:-2].split('\t')

freq_diffs_x = []
snp_list_x = []
freqs_x = []

#Read in the data. Freq is taken as the average over inland, or the average over coastal. Choice of populations should be confirmed.
for line in freq_file_x:
	split_line = line.decode()[0:-2].split('\t')
	av_A = np.average(np.array([split_line[6], split_line[7], split_line[9], split_line[10], split_line[11]], dtype = float)) #Coastal: Sumba_Kodi, Sumba_Lamboya, Sumba_Mamboro, Sumba_Rindi, Sumba_Wanokaka
	av_B = np.average(np.array([split_line[5], split_line[8], split_line[12]], dtype = float)) #Inland: Sumba_Anakalang, Sumba_Loli, Sumba_Wunga
	snp_list_x.append([split_line[0:5]])
	freqs_x.append(np.array(split_line[5:], dtype = float))
	freq_diffs_x.append([av_A, av_B])

freq_diffs_x = np.array(freq_diffs_x)
snp_list_x = np.array(snp_list_x)
freqs_x = np.array(freqs_x)


#Male freqs from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351883/
#16 / 417 = 0.038 #Inland
#45 / 427 = 0.105 #Coastal

print("\n\nCalculating empirical p-values\n\n")

print("Number of SNPs around 0.038 freq in inland", len(freq_diffs[(freq_diffs[::,1] > 0.033) * (freq_diffs[::,1] < 0.043)])) # +- 0.005
print("Number of these SNPs > 0.105 in coastal", np.sum(freq_diffs[(freq_diffs[::,1] > 0.033) * (freq_diffs[::,1] < 0.043)][::,0] > 0.105))
print("Empirical p val (auto)", np.sum(freq_diffs[(freq_diffs[::,1] > 0.033) * (freq_diffs[::,1] < 0.043)][::,0] > 0.105) / float(len(freq_diffs[(freq_diffs[::,1] > 0.033) * (freq_diffs[::,1] < 0.043)])))
##Empirical p val (auto) is 0.02142729065805989.

####Or on the X chromosome

print("Number of SNPs around 0.038 freq in inland (X)", len(freq_diffs_x[(freq_diffs_x[::,1] > 0.033) * (freq_diffs_x[::,1] < 0.043)])) # +- 0.005
print("Number of these SNPs > 0.105 in coastal (X)", np.sum(freq_diffs_x[(freq_diffs_x[::,1] > 0.033) * (freq_diffs_x[::,1] < 0.043)][::,0] > 0.105))
print("Empirical p val (X)", np.sum(freq_diffs_x[(freq_diffs_x[::,1] > 0.033) * (freq_diffs_x[::,1] < 0.043)][::,0] > 0.105) / float(len(freq_diffs_x[(freq_diffs_x[::,1] > 0.033) * (freq_diffs_x[::,1] < 0.043)])))
##Empirical p val (X) is 0.013888888888888888.


###Plotting:

def plot_empiricalp_autosomal(fig = None, index = 111, x_max = 0.2):
        autosomal_inland_bounds = freq_diffs[(freq_diffs[::,1] > 0.033) * (freq_diffs[::,1] < 0.043)]
        if fig is None:
                fig = pl.figure()
        ax = fig.add_subplot(index)
        ax.hist(autosomal_inland_bounds[::,0], bins = 50)
        ax.vlines(x = 0.105, ymin = 0.0, ymax = 800, colors = 'darkred', linestyles = 'dashed')
        ax.set_ylim(0,800)
        ax.set_xlim(0,x_max)
        ax.set_xlabel('Coastal derived allele frequency (autosome)')
        ax.set_ylabel('SNP count (autosome)')
        return ax

def plot_empiricalp_x(fig = None, index = 112, x_max = 0.2):
        x_inland_bounds = freq_diffs_x[(freq_diffs_x[::,1] > 0.033) * (freq_diffs_x[::,1] < 0.043)]
        if fig is None:
                fig = pl.figure()
        ax = fig.add_subplot(index)
        ax.hist(x_inland_bounds[::,0], bins = 25)
        ax.vlines(x = 0.105, ymin = 0.0, ymax = 30, colors = 'darkred', linestyles = 'dashed')
        ax.set_ylim(0,30)
        ax.set_xlim(0,x_max)
        ax.set_xlabel('Coastal derived allele frequency (X)')
        ax.set_ylabel('SNP count (X)')
        return ax

def plot_empiricalp_both(x_max = 0.2):
        fig = pl.figure()
        plot_empiricalp_autosomal(fig = fig, index = 211, x_max = x_max)
        plot_empiricalp_x(fig = fig, index = 212, x_max = x_max)
        fig.tight_layout()
        pl.show()

def plot_empiricalp_2d_autosomal():
        #pl.hist2d(x = freq_diffs[::,0], y = freq_diffs[::,1], bins = 61, range = [[0.0,0.25],[0.0,0.25]], norm=pl.mpl.colors.LogNorm())
        pl.hist2d(x = freq_diffs[::,0], y = freq_diffs[::,1], bins = 37, range = [[0.0,0.20],[0.0,0.20]], norm=pl.mpl.colors.LogNorm())
        pl.gca().set_facecolor('snow')
        # Create a Rectangle patch
        rect_SNPs_1 = pl.mpl.patches.Rectangle((-0.01,0.038-0.005), 0.115, 0.01, linewidth = 2, edgecolor='powderblue', facecolor = 'none', ls = 'dashed')
        rect_SNPs_2 = pl.mpl.patches.Rectangle((0.105,0.038-0.005), 0.25-0.105, 0.01, linewidth = 2, edgecolor='orangered', facecolor = 'none', ls = 'dashed')
        pl.vlines(x = 0.105, ymin = 0.0, ymax = 0.25, colors = 'darkred', linestyles = 'dashed')
        pl.hlines(y = 0.038, xmin = 0.0, xmax = 0.25, colors = 'darkgrey', linestyles = 'dashed')
        pl.gca().add_patch(rect_SNPs_1)
        pl.gca().add_patch(rect_SNPs_2)
        pl.xlabel('Coastal derived allele frequency')
        pl.ylabel('Inland derived allele frequency')
        pl.colorbar()
        pl.show()


def plot_empiricalp_2d_x():
        pl.hist2d(x = freq_diffs_x[::,0], y = freq_diffs_x[::,1], bins = 20, range = [[0.0,0.20],[0.0,0.20]], norm=pl.mpl.colors.LogNorm())
        pl.gca().set_facecolor('snow')
        # Create a Rectangle patch
        rect_SNPs_1 = pl.mpl.patches.Rectangle((-0.01,0.038-0.005), 0.115, 0.01, linewidth = 2, edgecolor='powderblue', facecolor = 'none', ls = 'dashed')
        rect_SNPs_2 = pl.mpl.patches.Rectangle((0.105,0.038-0.005), 0.25-0.105, 0.01, linewidth = 2, edgecolor='orangered', facecolor = 'none', ls = 'dashed')
        pl.vlines(x = 0.105, ymin = 0.0, ymax = 0.25, colors = 'darkgrey', linestyles = 'dashed')
        pl.hlines(y = 0.038, xmin = 0.0, xmax = 0.25, colors = 'darkgrey', linestyles = 'dashed')
        pl.gca().add_patch(rect_SNPs_1)
        pl.gca().add_patch(rect_SNPs_2)
        pl.xlabel('Coastal derived allele frequency')
        pl.ylabel('Inland derived allele frequency')
        pl.colorbar()
        pl.show()
