"""User interface"""
from electionMachine import electionMachine

def yleUi():
    while True:
        ask=input("""\n***** Parliament Election Result Predicter *****\n
        With this program, you can define candidate characteristics and \n
        simulate their chance of getting elected into the Finnish parliament.\n
        Data the model is built on comes from the elections 2015.\n
        Press...\n
        s to -> Start simulation\n
        q to quit """)

        if ask =="q":
            sure = input("Are you sure you want to quit? y/n")
            if sure == "y":
                break
            else:
                continue
        elif ask =="s":
            electionMachine()
        else:
            print("Unknown input")
            continue

