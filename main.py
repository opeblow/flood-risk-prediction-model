import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))

from src.train import train_model
from src.predict import predict_from_input
from src.utils import show_valid_options
def main():
    while True:
        print("\n FLOOD RISK SYSTEM")
        print("1 Train Model")
        print("2 Predict Flood")
        print("3 Exit")
        choice=input("Select an option (1/2/3):").strip()

        if choice=="1":
            print("nTraining the model..")
            train_model()
            print("Training Finished")

        elif choice=="2":
            print("\n Flood Risk Prediction Interface")
            show_valid_options()
            while True:
                try:
                    predict_from_input()
                    again=input("\n Predict again?(Yes or No):").strip().lower()

                    if again not in ["yes","y"]:
                        break
                except KeyboardInterrupt:
                    print("\nGoodBye!")
                    break
                except Exception as e:
                    print(f"\n Error:{e}\nPlease try again")

        elif choice=="3":
            print("Exiting")
            break

        else:
            print("Invalid Option.Please choose 1,2 or 3")

if __name__=="__main__":
    main()

                
