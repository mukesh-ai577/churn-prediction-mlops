import pandas as pd
import os   

def ingest_data(input_path , output_path) :
    # load raw data
    df = pd.read_csv(input_path)

    # convert TotalCharge to numeric

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'] , errors='coerce')

    # Fill missing TotalCharges
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Drop customerID
    df = df.drop("customerID", axis=1)


    # convert target into 0 and 1
    df['Churn'] = df['Churn'].map({"No" : 0 , "Yes" : 1})

    # create output directory if not exists

    os.makedirs(os.path.dirname(output_path) , exist_ok=True)

    # Save processed data
    df.to_csv(output_path , index=False)

    print("Data ingestion Completed.")
    print(f"processed data saved at : {output_path}")


if __name__ == "__main__" :
        input_path = "Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        output_path = "Data/processed/churn_processed.csv"
        ingest_data(input_path, output_path)
