import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from yle_model_input import yle_model
from questions import *

# Verify user input as integers
def inputNumber(message):
    while True:
        try:
            userInput = int(input(message))
        except ValueError:
            print("Not an integer! Try again.")
            continue
        else:
            return userInput

# Redefine election budget
def electionBudjet(electBudget):
    elect_budget_a = 0
    if electBudget == 0:
        elect_budget_a = 500
    elif electBudget == 1:
        elect_budget_a = 3000
    elif electBudget == 2:
        elect_budget_a = 7500
    elif electBudget == 3:
        elect_budget_a = 15000
    elif electBudget == 4:
        elect_budget_a = 35000
    elif electBudget == 5:
        elect_budget_a = 90000
    else:
        print('Unknown input: ', electBudget)
    return elect_budget_a

# Redefine yearly income
def yearly_income(yearly_income):
    yearly_income_a = 0
    if yearly_income == 0:
        yearly_income_a = 10000
    elif yearly_income == 1:
        yearly_income_a = 25000
    elif yearly_income == 2:
        yearly_income_a = 40000
    elif yearly_income == 3:
        yearly_income_a = 60000
    elif yearly_income == 4:
        yearly_income_a = 85000
    elif yearly_income == 5:
        yearly_income_a = 130000
    else:
        print('Unknown input: ', yearly_income)
    return yearly_income_a

# Obtain user input and save it in list
def getUserData():
    userInput = []
    # Defining user input
    age = inputNumber(ageQ)
    sex = inputNumber(sexQ)
    celeb = inputNumber(celebQ)
    parliam = inputNumber(parliamQ)
    motherT = inputNumber(motherTQ)
    lang = inputNumber(langQ)
    web = inputNumber(webQ)
    children = inputNumber(childrenQ)
    employer = inputNumber(employerQ)
    educ = inputNumber(educQ)
    work = inputNumber(workQ)
    pmParty = inputNumber(pmPartyQ)
    extFund = inputNumber(extFundQ)
    electBudget = electionBudjet(inputNumber(electBudjQ))
    y_income = yearly_income(inputNumber(yIncomeQ))

    # Adding user input to list
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
    userInput.append(electBudget)
    userInput.append(y_income)

    return userInput

 # Transform categorical columns to yes/no columns per column value
def oneHotEnc(df):
    OH_encoder1 = OneHotEncoder(handle_unknown='ignore', sparse=False)
    userInputToList = df.columns.tolist()
    userInputList = userInputToList[4:10]
    userInputList.remove('children')
    userInputList.remove('twitter_account')
    userInputDF_OH = pd.DataFrame(OH_encoder1.fit_transform(df[userInputList]))
    userInputDF_OH.index = df.index
    UserInputOld = df.drop(userInputList, axis=1)
    X_user_handled = pd.concat([UserInputOld, userInputDF_OH], axis=1)
    return X_user_handled

# Main function in file
def electionMachine():

    while True:

        print("""\n\nData needed for the prediction is the following:\n 
        age, sex, celebrity status, currently in parliament, mother tounge, langauges known, twitter account,
        children, employer, education, work role, election budget, yearly income, party status and external funding.\n""")
        userInput = getUserData()

        # Creating imaginary user inputs
        mylist0 = [32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 10000]
        mylist1 = [32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 500, 10000]
        mylist2 = [32, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 500, 10000]
        mylist3 = [32, 1, 1, 1, 3, 0, 1, 1, 3, 3, 0, 1, 1, 500, 10000]
        mylist4 = [32, 1, 1, 1, 1, 0, 1, 1, 2, 4, 0, 1, 1, 500, 10000]
        mylist5 = [32, 1, 1, 1, 1, 0, 1, 1, 0, 5, 0, 1, 1, 500, 10000]
        mylist6 = [32, 1, 1, 1, 1, 0, 1, 1, 1, 6, 0, 1, 1, 500, 10000]
        mylist7 = [32, 1, 1, 1, 1, 0, 1, 1, 1, 7, 0, 1, 1, 500, 10000]
        mylist8 = [32, 1, 1, 1, 1, 0, 1, 1, 1, 8, 0, 1, 1, 500, 10000]

        # Creating dataframe from real and fake user input to cover all possible values in columns
        userInputDf = pd.DataFrame([userInput,
                                    mylist0,
                                    mylist1,
                                    mylist2,
                                    mylist3,
                                    mylist4,
                                    mylist5,
                                    mylist6,
                                    mylist7,
                                    mylist8],
                                    columns=['age',
                                             'sex', 'celebrity', 'currently_in_parliament', 'education',
                                             'mother_tongue', 'twitter_account', 'children', 'employer',
                                             'work_status', 'languages_known', 'PM_party',
                                             'external_election_funding', 'elect_budget_', 'yearly_income_'])

        # Check that user gave the right input
        print("\n")
        print(userInputDf.iloc[0, :])
        answer = input("""\n Your input data is listed above.
        Election budget and yearly income shows the class midpoints in the classes you chose. 
        Is the data correct y/n?: """)
        if answer == "y":
            print("\n")
        else:
            oh = input('Oh, you have to start again! Press any button ')
            userInput.clear()
            continue

        # Process our user input with One Hot encoder: mother tougue, employer, education and work status
        userX = oneHotEnc(userInputDf)
        print('***** PREDICTION RESULT  *****\n')
        percent, modelInfo = yle_model(userX)
        print('Prediction to get elected is: %d percent ' % percent)

        finish = input('\nDo you want to make another prediction, y/n?')
        if finish == "y":
            userInput.clear()
            continue
        else:
            break
