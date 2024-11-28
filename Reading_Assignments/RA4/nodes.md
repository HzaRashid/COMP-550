# 1. How does this paper propose to measure gender bias in coreference resolution systems? Explain how an "unbiased" system should perform, versus the results obtained in the experiments.
The paper, "Gender Bias in Coreference Resolution" (Rudinger et al., 2018) proposes dataset schemas for measuring gender bias in Coreference Resolution Systems. To this end, the authors
focus on gender bias with respect to occupations, evaluating the accuracy of Rule-based, Statisical, and Neural Coreference Systems in resolving a pronoun (male, female, or neutral) to a coreferent antecedent that is either an occupation or a participant. They constructed a challenge dataset in the style of Winograd schemas, wherein a pronoun must be resolved to one of two previously mentioned entities in a sentence. 

list of 60 one-word occupations obtained from Caliskan et al. (2017) (see supplement), with corresponding gender percentages
available from the U.S. Bureau of Labor Statistics.4 For each occupation, we wrote two similar sentence templates: one in which PRONOUN is
coreferent with OCCUPATION, and one in which
it is coreferent with PARTICIPANT (see Figure 2).
For each sentence template, there are three PRONOUN instantiations (female, male, or neutral),
and two PARTICIPANT instantiations (a specific
participant, e.g., “the passenger,” and a generic
paricipant, “someone.”) With the templates fully
instantiated, the evaluation set contains 720 sentences: 60 occupations × 2 sentence templates per
occupation × 2 participants × 3 pronoun genders.


# 2. The paper discusses a potential case of bias amplification involving the occupation manager on page 4. Explain the proposed mechanism in which dataset bias could be amplified at the system level, then how system bias could then lead to further amplification in society downstream.
The authors found t