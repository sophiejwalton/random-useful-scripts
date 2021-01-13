import os
import glob
import copy
import numpy as np
import Bio
import scipy.spatial
import itertools as it 


def ecdf_vals(data):
    """Return x and y values for an ECDF."""
    return np.sort(data), np.arange(1, len(data)+1) / len(data)
def opposite_codon_frequencies(aminoacidweights, gencodeweights):
    opposite_aminoacidweights ={}
    opposite_gencodeweights = {}
    
    for key in aminoacidweights.keys():
        
        
        if len(aminoacidweights[key]) == 1.:
            opposite_aminoacidweights[key] = [1.]
        else: 
            opposite_aminoacidweights[key] = 1. - aminoacidweights[key]
    
    for key in gencodeweights.keys():
        
        if gencodeweights[key] == 1.:
            opposite_gencodeweights[key] = 1.
        else:
            opposite_gencodeweights[key] = 1. - gencodeweights[key]
            
    
    return opposite_aminoacidweights, opposite_gencodeweights

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

def get_codon_frequencies_doublets(transcriptome):
    '''
    Obtains all ordered codon doublet frequencies from transcriptome 
    
    transcriptome: a string of all transcripts of an organism concatenated
    
    returns: dictionary where keys are amino acids and values are frequencies of amino acid in
    dictionary 
    '''
    
    codon_counts_dic = {}
  
    for codons in it.permutations(gencode.keys(), 2):
        c1, c2 = codons
        codon_counts_dic[c1 + c2] = 0.
    for codon in gencode.keys():
        codon_counts_dic[codon + codon] = 0.
        
    
  
    ncodons = int(len(transcriptome)//3)
    ncodons_good=0
    for i in range(ncodons):
        start = i*3
        end = i*3 + 6
        if end > len(transcriptome):
            continue 
        codons = transcriptome[start:end].upper()
      #  print(codons)
        codon_counts_dic[codons] = codon_counts_dic[codons] + 1 
        ncodons_good = ncodons_good + 1

   
    codon_frequencies_dic = {}
    npairs = len(codon_counts_dic.keys())
  
    for codons in codon_counts_dic.keys():
        codon_frequencies_dic[codons] = codon_counts_dic[codons]/ncodons_good
    return codon_frequencies_dic



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
    
def random_reverse_translate(protein_seq, weights_dic = None, change_seq = '', random = True, wiggle = False,
                                       doubletscode = {}, pairs = True):
    """
    Return a random cDNA sequence for a given protein sequence (protein_seq). 
    """
    cDNA_seq = ''
    weights = np.zeros(1)
    prev_codon = ''
    if len(doubletscode) == 0:
        pairs = False
    for i, aminoacid in enumerate(protein_seq):
        if not pairs:
            prev_codon = '' # do not incorporate information about previous codon
        if weights_dic:
            weights = weights_dic[aminoacid]
        if len(change_seq) > 0:
            random_codon = get_random_codon(aminoacid, doubletscode = doubletscode, random = random, wiggle = wiggle,
                                               weights = weights, prev_codon = prev_codon,
                                                   dont_use = change_seq[3*i:3*(i+1)])
            cDNA_seq = cDNA_seq + random_codon
            prev_codon = random_codon
            
        else:
            random_codon = get_random_codon(aminoacid, doubletscode = doubletscode, random = random, wiggle = wiggle,
                                               weights = weights, prev_codon = prev_codon,
                                                   )
            cDNA_seq = cDNA_seq + random_codon
            prev_codon = random_codon
            
        
    assert translate(cDNA_seq) == protein_seq
    return cDNA_seq

def get_codon_frequencies(transcriptome):
    '''
    Obtains codon frequencies from transcriptome 
    
    transcriptome: a string of all transcripts of an organism concatenated
    
    returns: dictionary where keys are amino acids and values are frequencies of amino acid in
    dictionary 
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




def get_codon_weights(codon_frequencies_dic, aminoacidcode = aminoacidcode):
    '''Creates a codon weighths for each codon from a dictionary of codon frequencies. 
    weights are computed as follows: w_i = x_i,j/yj where x_i,j is the frequency of codon i for amino acid j
    in the transcriptome and y_j is the frequency of the maximally frequent codon for amino acid j 
    
    inputs:
        frequencies: a dictionary of the codon frequencies (values) for each amino acid (keys)
    
    outputs:
        
        aminoacidweights: keys are amino acids, values are arrays of w_i for all synonymous codons. The order of the codons is the as those used in aminoacidcode. 
    
        gencodeweights: keys are codons, values are w_i for each codon
    '''
    

    
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
    '''
    Obtains Codon Adaptation Index (CAI) for a given DNA_seq calculated using weights_dic
    CAI = (w_1*.w_i*..w_N)^(1/N) where w_i is the weight of codon i. 
    
    Inputs:
        dna_seq: ORF in form of string to evaluate CAI
        weights_dic: dictionary of CAI weights for each codon. Values are weights and keys are codons. 
    '''
    if len(dna_seq) % 3 > 0.:
        raise ValueError("Length of DNA sequence must be divisble by 3")
    ncodons = int(len(dna_seq)//3)
    cai = 1
    for i in range(ncodons):
        codon = dna_seq[i*3:(i+1)*3].upper()
        cai = cai*weights_dic[codon]
    return cai**(1/ncodons)

def get_doublest_likelihood(dna_seq, weights_dic):
    '''
    Obtains Codon Adaptation Index (CAI) for a given DNA_seq calculated using weights_dic
    CAI = (w_1*.w_i*..w_N)^(1/N) where w_i is the weight of codon i. 
    
    Inputs:
        dna_seq: ORF in form of string to evaluate CAI
        weights_dic: dictionary of CAI weights for each codon. Values are weights and keys are codons. 
    '''
    if len(dna_seq) % 3 > 0.:
        raise ValueError("Length of DNA sequence must be divisble by 3")
    ncodons = int(len(dna_seq)//3)
    score = 1
    for i in range(ncodons-1):
        start = i*3
        end = start + 6
        codonpair = dna_seq[start:end].upper()
        score = score*weights_dic[codonpair]
    
    return score

def get_hamming_dist(seq1, seq2, normalize = True):
    '''
    Calculates hamming distance (number of positions where seq1 != seq2) for two equal length strings (seq1 and seq2).
    If Normalize = True, return hamming distance / length(seq1) 
    '''
    if len(seq1) != len(seq2):
        print(len(seq1), len(seq2))
      #  raise ValueError("sequence lengths must be equal")
    dist = 0.
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            dist += 1
    if normalize:
        return dist/len(seq1)
    return dist
        

def get_rnai_score(dna_seq1, dna_seq2, weights_dic, weight_distance = .5):
    '''
    For two DNA sequences and a dictionary of codon weights (key: codon, value: weight) calculates the 
    CAI and Normalized Hamming distance. Performs a weighted average to combine the scores. 
    
    Returns CAI, Dist, Combined_score
    '''
    cai = get_CAI(dna_seq1, weights_dic)
    dist = get_hamming_dist(dna_seq1, dna_seq2, normalize = True)
    combined_score = (1-weight_distance)*cai + weight_distance*dist
    return cai, dist, combined_score 




def get_random_codon(aminoacid, aminoacidcode = aminoacidcode, doubletscode = {}, random = True, wiggle = False,
                     dont_use = '', weights = np.zeros(1),
                    prev_codon = ''):
    '''Gets a random codon for aminoacid, aminoacid must be single letter amino acid code.
    '''
    
    codon_list = aminoacidcode[aminoacid]
    weights = np.copy(weights)
    doubletscode = copy.copy(doubletscode)

    if sum(weights == 0):
        weights = np.ones(len(codon_list)) # uniformly distributed weights list 
    if len(codon_list) < 2:
        dont_use = '' # only one option 
        
        
    if dont_use != '':
        ind = codon_list.index(dont_use.upper())
        
        if len(codon_list) == 2 and wiggle and prev_codon == '':
            extra_weight = .25
            weights[ind] = weights[ind]*extra_weight
            
        else:
            if ind + 1 == len(codon_list):
                codon_list = codon_list[:ind]
                weights = np.array(list(weights[:ind]))
            else:
                codon_list = codon_list[:ind] + codon_list[ind + 1:]
                weights = np.array(list(weights[:ind])  + list(weights[ind + 1:]))  
    if prev_codon != '':
        doublets = []
        weights = []
        for codon in codon_list:
            doublets.append(prev_codon + codon)
            #print(doublets)
            weights.append(doubletscode[prev_codon + codon])
        if (len(codon_list) == 1) and dont_use != '':
            if wiggle:
                extra_weight = .25
            else:
                extra_weight = 0.
            codon_list.append(dont_use)
            doublets.append(prev_codon + dont_use)
            weights.append(doubletscode[prev_codon + dont_use]*extra_weight) # sometimes alternative is terrible, so this allows for more wiggleroom
          #  if wiggle:
              #  print(codon_list)
           
       # print(codon_list, doublets, weights)
    if random:
       # print(codon_list, fweights)
        return np.random.choice(codon_list, p = np.array(weights)/np.sum(weights))
    else:
        if len(prev_codon) > 1.:
            print(doublets, weights)
        else:
            print(codon_list, weights)
        best_ind = np.argmax(weights)
        return codon_list[best_ind]


def get_RNAi_seq(ORF, protein, aminoacidweights, gencode_weights, trials = 1000, doubletscode = {}, pairs = True, random = True,
                 enforce_different_codons = False, wiggle = False):
    '''
    Obtains random samples of recoded ORFs that encodes for a protein sequence (protein) based on CAI weights. Evaluates CAI and hamming distance to original ORF. 
    
    Arguments:
        ORF: DNA sequence to be recoded 
        Protein: Protein sequence (string with single letter amino acid code)
                aminoacidweights: keys are amino acids, values are arrays of w_i for all synonymous codons. The order of the codons is the as those used in aminoacidcode. 
        gencodeweights: keys are codons, values are w_i for each codon
        trials: number of random sequences to generate (default 1000)
        enforce_different_codons: enforce that for all amino acids withh multiple codons that selected codons be different from the one used in the ORF at that position. 
    
    Returns: 
        seqs: Random sequences generate
        scores: average of CAI and hamming distance for each sequence
        cais: CAIs for each sequence
        dists: Hamming distance for each sequence
        
    '''
    seqs = []
    scores = []
    cais = []
    dists = []
    protein = protein.upper()
    if len(doubletscode) == 0:
        pairs = False
    for i in range(trials):
        if enforce_different_codons:
            random_seq = random_reverse_translate(protein, random = random,
                                          weights_dic = aminoacidweights, pairs = pairs, doubletscode = doubletscode, wiggle = wiggle,
                                             change_seq = ORF)
        else:
            random_seq = random_reverse_translate(protein, random = random, pairs = pairs, doubletscode = doubletscode,
                                                  wiggle = wiggle,
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