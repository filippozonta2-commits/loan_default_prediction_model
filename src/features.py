FINAL_STATUSES = {"Fully Paid": 0, "Does not meet the credit policy. Status:Fully Paid": 0, "Charged Off": 1, "Default": 1, "Does not meet the credit policy. Status:Charged Off": 1}
FEATURES = ["loan_amnt", "term", "int_rate", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
NUMERIC_FEATURES = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
CATEGORICAL_FEATURES = ["term", "grade", "emp_length", "home_ownership", "verification_status", "purpose"]
LEAKAGE_COLUMNS = ["out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d"]
