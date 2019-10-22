# election_predicter
This application predicts a candidates chance of getting elected into the Finnish national parliament.

The data used for the prediction comes from avoindata.fi and consists of parliament candidates answers to yle's election machine 2015.

All files but yle.py are part of the user application, where one can try different background factors for
parliament candidates and see their predicted chances of getting elected. yle.py is a file I have used to
test different prediction options and algorithms as well as different ways of wrangling the data which the
prediction model is built upon. 

Information on variable definitions:
Celebrity definition:
https://www.is.fi/vaalit2015/art-2000000912037.html & https://www.iltalehti.fi/uutiset/a/2015012419074122

Education level definition:
elementary school = grade 1-9, vocational or matriculation exam = grade 10-12, university level = grade 13 and upwards
