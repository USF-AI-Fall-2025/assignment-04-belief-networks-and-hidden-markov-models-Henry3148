# Belief-Networks-Hidden-Markov-Models
## Fall 2025 CS 362/562
## Author: Henry Olson

This program trains a hidden markov based viterbi algorithm to take user text and correct it. The program has been trained with custom spelling erorrs and works for around 90% of tests. 

## Reflection

*1. Give an example of a word which was correctly spelled by the user, but which was incorrectly
“corrected” by the algorithm. Why did this happen?*

The model performed poorly when given correctly spelled words. For example it changed "movies" to "moving". This is likely due to the small lexicon and training data. This can cause scoring for certain changes to actually be higher than the correct word itself. It is also likely bias to the words included int the training data since it trains for "moving" but never "movies".

*2. Give an example of a word which was incorrectly spelled by the user, but which was still
incorrectly “corrected” by the algorithm. Why did this happen?*

There was a correction from "happly" to "happily" when I was intending for it to correct to "happy". This is a small issue but shows precedence that insertions have over deletions. I also noticed that longer and more complex mispellings like "happtlye" didn't get corrected at all and were to complex to have a decent score.

*3. Give an example of a word which was incorrectly spelled by the user, and was correctly corrected
by the algorithm. Why was this one correctly corrected, while the previous two were not?*

I was surprised by how well the model handeled combinations of "believe". For "believs" and "beliebe" the model output the correct spelling. I think this word performed well becase it prioritized the fast, same size viterbi which works very well and didnt complicate it with insertions and deletions. 

*4. How might the overall algorithm’s performance differ in the “real world” if that training dataset is
taken from real typos collected from the internet, versus synthetic typos (programmatically
generated)?*

This is a big issue and is evident with the data how much there is to improve. With actual real scenerio typos there will be humanistic patterns that our models can match better to. For example there is a huge pattern with mispellings due to keyboard slipups and key orientation. There could also be the consideration of more phonetically common misspellings and duplicate letters. Training on a much larger "real" typo dataset would definitely yield a more comprehensive and accurate program. 
