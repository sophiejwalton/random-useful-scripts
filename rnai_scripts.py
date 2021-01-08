import os
import glob
import copy
import numpy as np
import Bio
import scipy.spatial

# create dictionaries of amino acids and codons 
gencode = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'}



aminoacidcode = {'I': ['ATA', 'ATC', 'ATT'],
 'M': ['ATG'],
 'T': ['ACA', 'ACC', 'ACG', 'ACT'],
 'N': ['AAC', 'AAT'],
 'K': ['AAA', 'AAG'],
 'S': ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT'],
 'R': ['AGA', 'AGG', 'CGA', 'CGC', 'CGG', 'CGT'],
 'L': ['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG'],
 'P': ['CCA', 'CCC', 'CCG', 'CCT'],
 'H': ['CAC', 'CAT'],
 'Q': ['CAA', 'CAG'],
 'V': ['GTA', 'GTC', 'GTG', 'GTT'],
 'A': ['GCA', 'GCC', 'GCG', 'GCT'],
 'D': ['GAC', 'GAT'],
 'E': ['GAA', 'GAG'],
 'G': ['GGA', 'GGC', 'GGG', 'GGT'],
 'F': ['TTC', 'TTT'],
 'Y': ['TAC', 'TAT'],
 '*': ['TAA', 'TAG', 'TGA'],
 'C': ['TGC', 'TGT'],
 'W': ['TGG']}

def read_fasta(filename):
    """Read a sequence in from a FASTA file containing a single sequence.

    We assume that the first line of the file is the descriptor and all
    subsequent lines are sequence.
    
    This function from solution 4.1 of Caltech Bi 203 bootcamp 
    """
    with open(filename, 'r') as f:
        # Read in descriptor
        descriptor = f.readline().rstrip()

        # Read in sequence, stripping the whitespace from each line
        seq = ''
        line = f.readline().rstrip()
        while line != '':
            seq += line
            line = f.readline().rstrip()

    return descriptor, seq

def translate(dna_seq):
    """
    Translates an ORF to a protein (must be DNA) 
    """
    dna_seq = dna_seq.upper()
    ncodons = int(len(dna_seq)/3)
 
    if len(dna_seq) % 3 > 0.:
        raise ValueError("Length of DNA sequence must be divisble by 3")
        
    protein_seq = ''
    for i in range(ncodons):
        protein_seq = protein_seq + gencode[dna_seq[i*3:(i+1)*3]]
    return protein_seq
    
def random_reverse_translate(protein_seq, weights_dic = None, change_seq = ''):
    """
    Return a random cDNA sequence for a given protein sequence (protein_seq). 
    """
    cDNA_seq = ''
    weights = np.zeros(1)
    for i, aminoacid in enumerate(protein_seq):
        if weights_dic:
            weights = weights_dic[aminoacid]
        if len(change_seq) > 0:
            cDNA_seq = cDNA_seq + get_random_codon(aminoacid, 
                                               weights = weights, dont_use = change_seq[3*i:3*(i+1)])
        else:
            cDNA_seq = cDNA_seq + get_random_codon(aminoacid, weights = weights)
            
        
    assert translate(cDNA_seq) == protein_seq
    return cDNA_seq

def get_codon_frequencies(transcriptome):
    '''
    Obtains codon frequencies from transcriptome 
    
    transcriptome: a string of all transcripts of an organism concatenated
    '''
    
    codon_counts_dic = {}
    for codon in gencode.keys():
        codon_counts_dic[codon] = 0.
        
    
    ncodons = int(len(transcriptome)//3)
    for i in range(ncodons):
        codon = transcriptome[i*3:(i+1)*3].upper()
  
        codon_counts_dic[codon] = codon_counts_dic[codon] + 1 
    codon_frequencies_dic = {}
    for codon in codon_counts_dic.keys():
        codon_frequencies_dic[codon] = codon_counts_dic[codon]/ncodons
    return codon_frequencies_dic


def get_codon_weights_dic(transcriptome):
    '''Creates a codon weighths for each codon from a given transcriptome 
    weights are computed as follows: w_i,j = x_i,j/yj where x_i,j is the frequency of codon i for amino acid j
    in the transcriptome and y_j is the frequency of the maximally frequent codon for amino acid j 
    
    transcriptome: a string of all transcripts of an organism concatenated
    '''
    
    codon_frequencies_dic = get_codon_frequencies(transcriptome)
    
    aminoacid_weights_dic = {}
    gencode_weights_dic = {}
    
    for aminoacid in aminoacidcode.keys():
        codons = aminoacidcode[aminoacid]
        codon_weight_array = np.zeros(len(codons))
        codon_freq_array = np.zeros(len(codons))
        y_j = 0 
        for i, c in enumerate(codons):
            freq = codon_frequencies_dic[c]
            codon_freq_array[i] = freq
            if freq > y_j:
                y_j = freq
        codon_weight_array = codon_freq_array/y_j 
        aminoacid_weights_dic[aminoacid] = codon_weight_array
        for i, c in enumerate(codons):
            gencode_weights_dic[c] = codon_weight_array[i]
    return aminoacid_weights_dic, gencode_weights_dic 

def get_CAI(dna_seq, weights_dic):
    if len(dna_seq) % 3 > 0.:
        raise ValueError("Length of DNA sequence must be divisble by 3")
    ncodons = int(len(dna_seq)//3)
    cai = 1
    for i in range(ncodons):
        codon = dna_seq[i*3:(i+1)*3].upper()
        cai = cai*weights_dic[codon]
    return cai**(1/ncodons)


def get_hamming_dist(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("sequence lengths must be equal")
    dist = 0.
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            dist += 1
    return dist
        
def get_distance(seq1, seq2):
    
    return get_hamming_dist(seq1, seq2)/len(seq1)


def get_rnai_score(dna_seq1, dna_seq2, weights_dic, weight_distance = .5):
    cai = get_CAI(dna_seq1, weights_dic)
    dist = get_distance(dna_seq1, dna_seq2)
    combined_score = (1-weight_distance)*cai + weight_distance*dist
    return cai, dist, combined_score 

def get_random_codon(aminoacid, aminoacidcode = aminoacidcode, dont_use = '', weights = np.zeros(1)):
    '''Gets a random codon for aminoacid, aminoacid must be single letter amino acid code.
    '''
    
    codon_list = aminoacidcode[aminoacid]

    if sum(weights == 0):
        weights = np.ones(len(codon_list)) # uniformly distributed weights list 
    if len(codon_list) < 2:
        dont_use = '' # only one option 
        
    if dont_use != '':
        ind = codon_list.index(dont_use.upper())
        if ind + 1 == len(codon_list):
            codon_list = codon_list[:ind]
            weights = np.array(list(weights[:ind]))
        else:
            codon_list = codon_list[:ind] + codon_list[ind + 1:]
            weights = np.array(list(weights[:ind])  + list(weights[ind + 1:]))   

    return np.random.choice(codon_list, p = weights/np.sum(weights))


def get_RNAi_seq(ORF, protein, aminoacidweights, gencode_weights, trials = 10000, 
                 enforce_different_codons = False):
    '''
    '''
    seqs = []
    scores = []
    cais = []
    dists = []
    for i in range(trials):
        if enforce_different_codons:
            random_seq = random_reverse_translate(protein,
                                          weights_dic = aminoacidweights, 
                                             change_seq = ORF)
        else:
            random_seq = random_reverse_translate(protein,
                                          weights_dic = aminoacidweights)
        seqs.append(random_seq)
        cai, dist, score = get_rnai_score(random_seq, ORF, 
                           gencode_weights, weight_distance = .5)
        scores.append(score)
        cais.append(cai)
        dists.append(dist)
        
    return seqs, scores, cais, dists


def read_many_fasta(filename):
    """Read a sequence in from a FASTA file containing a single sequence.

    We assume that the first line of the file is the descriptor and all
    subsequent lines are sequence.
    
    This function from solution 4.1 of Caltech Bi 203 bootcamp 
    """
    descriptors = []
    seqs = []
    current_seq = ''
    end = False
    it = 0
    with open(filename, 'r') as f:
        while not end:
            line  = f.readline().rstrip()
            if line == '':
                end = True
                seqs.append(current_seq)
            elif '>' in line:
                if len(descriptors) >= 1:
                 
                    seqs.append(current_seq)

                descriptors.append(line)
                current_seq = ''
                
            else:
                current_seq = current_seq + line
            
    return descriptors, seqs