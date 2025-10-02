# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.5c3153xm9mha).

## Part 1
* Model accuracy: Dev - 0.9324618736383442
                  Test - 0.9331896551724138
* Free response:

There are a lot of parts of this that seem to work really well. Most of the words are classified correctly overall. First of all, the numbers and times are right. Plurals vs singular nouns also seem to be pretty solid. Function/filler words (like ‘the’, ‘from’, ‘to’, ‘would’, etc) are also very good. Proper nouns that are made up of multiple words (“New York”, “Los Angeles”, etc) are also turning out pretty well. 
In [02], the noun “airport” is marked as a verb. A few verb forms are also off, marked VBP when they should be VB. Things correctly or incorrectly tagged WDT also seem to be messed up sometimes. 

When words are tagged incorrectly, I think it ends up being because the model has not memorized that pattern yet. I think the more common tags are a better default for the model when it is unsure, so those end up showing up more in incorrect classifications. Overall, more training data or more epochs could help it be able to recognize certain patterns better. 

I think the micro-errors happen more frequently, like with the verb forms, but are less detrimental than the macro-errors. Macro-errors happen less often but are more of a problem when they do occur. 


## Part 2
* How many unique rules are there? 394
* What are the top five most frequent rules, and how many times did each occur?
Top 5 most frequent rules:
NNP -> NNP # 860
IN -> IN # 483
PUNC -> PUNC # 469
NN -> NN # 335
NNS -> NNS # 278

* What are the top five highest-probability rules with left-hand side NNP, and what are their probabilities?
Top 5 NP rules by probability:
NP -> NNP NNP # 0.1928
NP -> NP NP* # 0.1159
NP -> DT NN # 0.1014
NP -> DT NNS # 0.0855
NP -> DT NP* # 0.0841

* Free Response: Did the most frequent rules surprise you? Why or why not?
The most frequent rules are not super surprising to me. Since these word-level rules (like NNP -> NNP) show up once every token, the “most frequent rules” will just be a reflection of which POS tags are the most frequent in the data. 


## Part 3
* CKY accuracy using gold POS tags: 90%
* CKY accuracy using predicted POS tags: 60%
* Free response:
There is a slight difference in which sentences it fails to parse given gold tags vs the output from my tagger. For the gold tags, the only one that failed was “What airline is this?”, while with the predicted tags there were a few more that failed. This is because of tagger errors that block the necessary rules, like mixups with verb forms and proper nouns. 
It did well on #1, 3, 6, 7, 8, and 10. These ones use very common patterns which the grammar already has strong counts for, so even if it has a little noise the CKY parser still does a good job with them. 
One that it parsed awkwardly was #5. It parsed but kind of had an odd structure, with split letters and weird NP* chains. I think this is because sometimes handling proper names can be weird or inconsistencies in general lead it to output a phrase that is technically legal but not very clean. 

