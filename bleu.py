
from nltk.translate.bleu_score import sentence_bleu
ref_sent = 'Three  hundred  five copies of the Manual Of Macropathological Techniques were distributed.'
tgt_sent = "We gave 350 manuals."
reference = [ref_sent.split(' '), tgt_sent.split(' ')]
candidate = 'macropathological techniques were distributed.'.split(' ')
score = sentence_bleu(reference, candidate)
print(score)