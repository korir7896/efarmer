Resequencing data for one diploid population
============================================

Files in this directory
-----------------------

variants.vcf.gz
    Phased, biallelic single-nucleotide variants for 40 diploid individuals
    (80 haplotypes), named IND001 .. IND040.

    Five chromosomes, chr1 .. chr5, each 14,000,000 bp. Positions are
    1-based within their chromosome.

    REF is the ANCESTRAL allele and ALT the DERIVED allele at every site; the
    INFO field repeats the ancestral allele as AA. Allele 1 in the genotype
    column is therefore always the derived allele, and the unfolded site
    frequency spectrum can be read directly without further polarization.

    Only sites that are biallelic and segregating within these 40 individuals
    are present. Sites that are monomorphic in the sample are not listed.
    There are no missing genotypes.

recombination_map.bed
    Crossover rate per base pair per generation, as a tab-separated BED-like
    table with a header line: chrom, start, end, rate. Intervals are
    half-open [start, end), 0-based, and together they tile each chromosome
    exactly once. The rate is piecewise constant and is NOT uniform along a
    chromosome.

    Chromosomes assort independently of one another.

Parameters you may take as known
--------------------------------

Mutation rate
    2.0e-8 per base pair per generation, constant along the genome and
    constant over time.

    Without this the absolute effective population sizes and the absolute
    times in generations are not identifiable from the data; only products
    such as N*mu are. It is given for the same reason a real study takes a
    per-generation rate from the literature.

Selection
    Where a variant is under positive selection, fitness is ADDITIVE
    (co-dominant) in the number of derived copies: the three genotype
    fitnesses are 1, 1 + s and 1 + 2s for zero, one and two copies of the
    selected allele. The selection coefficient s is therefore the advantage
    per copy.

    This convention matters: the same sweep described under a different
    dominance parameterization implies a different numerical s.

Ploidy and mating
    Diploid, randomly mating, one panmictic population throughout. No
    population structure, no migration, no admixture, no inbreeding, no
    selfing, and no sex chromosomes: all five chromosomes are autosomes.

Generation time
    Irrelevant. Report every time in generations before the present, where
    the present is the moment the sample was taken.

What is not in the data
-----------------------

Causal sites are not marked, and a beneficial variant is not guaranteed to
appear in the VCF at all. A variant that reached fixation in the population
is monomorphic in the sample and so is absent by the filtering rule above.
Sweeps have to be found from their effect on linked neutral variation.
