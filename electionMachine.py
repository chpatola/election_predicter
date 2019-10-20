from yle_model_input import yle_model
import warnings
warnings.filterwarnings("ignore")

#Verify user input as integers
def inputNumber(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Try again.")
       continue
    else:
       return userInput 
        
#Main function in file
def electionMachine():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    from pandas.plotting import scatter_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import OneHotEncoder

    #List of user answers to questions for prediction
    userInput = []

    #Questions for prediction
    ageQ = "Age in years, for example 32: "
    sexQ ="Sex: female = 0, man = 1 : "
    celebQ ="Celebrity: 0 = no, 1 = yes: "
    parliamQ="Currently in parliament: 0 = no, 1 = yes: "
    motherTQ="""Mother toungue:
    other than finnish or swedish = 0
    swedish = 1
    finnish = 2: """
    langQ="Number of languages candidate knows: "
    webQ="Has twitter : 0 = no, 1 = yes: "
    childrenQ="Has children: no =0, yes = 1: "
    employerQ="""Employer:
    not working = 0
    public employer = 1
    other than public, private or not working = 2
    private employer = 3: """
    educQ = """Education level:
    0 = elementary school
    1 = vocational school or similar
    2 = university / university of applied sciences
    3 = other: """
    workQ="""Work status: expert/public officer/higly educated = 0
    pensioner = 1
    management/leading position = 2
    other than included on this list = 3
    farmer = 4
    student = 5
    artist = 6
    worker = 7
    entreprenour = 8: """
    pmPartyQ="Member in current prime minister party, 0 = no, 1 = yes: "
    extFundQ="Has external funding for election, 0 = no, 1 = yes: "
    electBudjQ="""Election budget:
    <1k € = 0
    1-5k € = 1
    5-10 k € = 2
    10-20k € =3
    20-50k € = 4
    >50k € = 5: """
    yIncomeQ="""Yearly income: <20k € = 0
    20-30k € = 1
    30-50k € = 2
    50-70k € = 3
    70-100k € = 4
    >100k € = 5: """

    #Main subfunction in the file
    while True:
    
        print("""\n\nData needed for the prediction is the following:\n 
        age, sex, celebrity status, currently in parliament, mother tounge, langauges known, twitter account,
        children, employer, education, work role, election budget, yearly income, party status and external funding.\n""")
        
        #Defining user input
        age =inputNumber(ageQ)
        sex= inputNumber(sexQ)
        celeb = inputNumber(celebQ)
        parliam= inputNumber(parliamQ)
        motherT = inputNumber(motherTQ)
        lang = inputNumber(langQ)
        web = inputNumber(webQ)
        children =inputNumber(childrenQ)
        employer =inputNumber(employerQ)
        educ =inputNumber(educQ)
        work =inputNumber(workQ)
        pmParty=inputNumber(pmPartyQ)
        extFund = inputNumber(extFundQ)
        electBudget =inputNumber(electBudjQ)
        elect_budget_a =0
        if electBudget == 0:
            elect_budget_a = 500
        elif electBudget ==1:
            elect_budget_a =3000
        elif electBudget ==2:
            elect_budget_a =7500
        elif electBudget ==3:
            elect_budget_a =15000      
        elif electBudget ==4:
            elect_budget_a =35000
        elif electBudget ==5:
            elect_budget_a =90000
        else:
            print('Unknown input: ',electBudget) 
        yearly_income =inputNumber(yIncomeQ)
        yearly_income_a = 0
        if yearly_income == 0:
            yearly_income_a = 10000
        elif yearly_income == 1:
            yearly_income_a =25000
        elif yearly_income == 2:
            yearly_income_a =40000
        elif yearly_income ==3:
            yearly_income_a =60000
        elif yearly_income == 4:
            yearly_income_a =85000
        elif yearly_income == 5:
            yearly_income_a =130000
        else:
            print('Unknown input: ',yearly_income)

        #Adding user input to list
        userInput.append(age)
        userInput.append(sex)
        userInput.append(celeb)
        userInput.append(parliam)
        userInput.append(educ)
        userInput.append(motherT)
        userInput.append(web)
        userInput.append(children)
        userInput.append(employer)
        userInput.append(work)
        userInput.append(lang)
        userInput.append(pmParty)
        userInput.append(extFund)
        userInput.append(elect_budget_a)
        userInput.append(yearly_income_a)

        #Creating imaginary user inputs
        mylist0 = [32,0,0,0,0,0,0,0,0,0,0,0,0,500,10000]
        mylist1 = [32,1,1,1,1,1,1,1,1,1,1,1,1,500,10000]
        mylist2 = [32,1,1,1,2,2,1,1,2,2,2,1,1,500,10000]
        mylist3 = [32,1,1,1,3,0,1,1,3,3,0,1,1,500,10000]
        mylist4 = [32,1,1,1,1,0,1,1,2,4,0,1,1,500,10000]
        mylist5 = [32,1,1,1,1,0,1,1,0,5,0,1,1,500,10000]
        mylist6 = [32,1,1,1,1,0,1,1,1,6,0,1,1,500,10000]
        mylist7 = [32,1,1,1,1,0,1,1,1,7,0,1,1,500,10000]
        mylist8 = [32,1,1,1,1,0,1,1,1,8,0,1,1,500,10000]

        #Creating dataframe from real and fake user input to cover all possible values in columns
        userInputDf = pd.DataFrame([userInput,mylist0, mylist1,mylist2,mylist3,mylist4,mylist5,mylist6,mylist7,mylist8],columns =['age',
        'sex', 'celebrity', 'currently_in_parliament', 'education',
       'mother_tongue', 'twitter_account', 'children', 'employer',
       'work_status', 'languages_known', 'PM_party',
       'external_election_funding', 'elect_budget_', 'yearly_income_'])
        
        #Check that user gave the right input
        print("\n")
        print(userInputDf.iloc[0,:])      
        answer = input("""\n Your input data is listed above.
        Election budget and yearly income shows the class midpoints in the classes you chose. 
        Is the data correct y/n?: """)
        if answer == "y":
            print("\n")
            
        else:
            oh = input('Oh, you have to start again! Press any button ')
            userInput.clear()
            continue
    
        #Process our user input with One Hot encoder: mother tougue, employer, education and work status
        
        OH_encoder1 = OneHotEncoder(handle_unknown='ignore', sparse=False)
        userInputToList = userInputDf.columns.tolist()
        userInputList = userInputToList[4:10]
        userInputList.remove('children')
        userInputList.remove('twitter_account')
        userInputDF_OH = pd.DataFrame(OH_encoder1.fit_transform(userInputDf[userInputList]))
        userInputDF_OH.index = userInputDf.index
        UserInputOld = userInputDf.drop(userInputList, axis=1)
        X_user_handled = pd.concat([UserInputOld, userInputDF_OH], axis=1)
        
        print('***** PREDICTION RESULT  *****\n' )
        percent = yle_model(X_user_handled)
        print('Prediction to get elected is: %d percent '%percent )
        
        finish = input('\nDo you want to make another prediction, y/n?')
        if finish == "y":
            userInput.clear()
            continue     
        else:
            break
            
       
        
