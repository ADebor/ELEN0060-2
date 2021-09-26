import numpy as np
import math
import pandas as pd

"""
ELEN0060 - Information and coding theory
Project 1 - Information measures

Implementation and medical diagnosis

Antoine DEBOR & Antoine DECKERS
"""

# ******************************************************************************
""" CONVENTION NOTE : JOINT TABLE P_{X,Y} WRITTEN AS line_j = Y_j AND column_i = X_i """
# ******************************************************************************

# --- MEASURES IMPLEMENTATION ---

def entropy(marginal_p_distrib):
    """
    Computes H(X) from P_X
    """

    H = 0
    for p in marginal_p_distrib:
        if p != 0:
            H -= p*math.log(p, 2)
    return H

def joint_entropy(joint_p_distrib):
    """
    Computes H(X,Y) from P_{X,Y}
    """

    H = 0
    n = np.shape(joint_p_distrib)[1]
    m = np.shape(joint_p_distrib)[0]
    for i in range(n):
        for j in range(m):
            arg = joint_p_distrib[j][i]
            if(arg !=0):
                H -= arg*math.log(arg, 2)
    return H

def conditional_entropy(conditional_p_distrib, marginal_p_distrib):
    """
    Computes H(X|Y) from P_{X|Y} and P_Y
    """

    # Computation of H(X|Y_j) for all Y_j
    H_j = []
    n = np.shape(conditional_p_distrib)[1]
    m = np.shape(conditional_p_distrib)[0]
    for j in range(m):
        H = 0
        for i in range(n):
            arg = conditional_p_distrib[j][i]
            if arg != 0:
                H -= arg * math.log(arg, 2)
        H_j.append(H.copy())
    # Computation of H(X|Y)
    H = 0
    for j, p in enumerate(marginal_p_distrib):
        H += p * H_j[j]
    return H

def mutual_information(marginal_p_distrib_X, marginal_p_distrib_Y, joint_p_distrib):
    """
    Computes I(X;Y) from P_X and P_Y and P_{X,Y}
    """

    I = 0
    n = len(marginal_p_distrib_X)
    m = len(marginal_p_distrib_Y)
    for i in range(n):
        for j in range(m):
            if(joint_p_distrib[j][i] != 0
            and marginal_p_distrib_X[i] != 0
            and marginal_p_distrib_Y[j] != 0):
                arg = (joint_p_distrib[j][i])/(marginal_p_distrib_X[i] * marginal_p_distrib_Y[j])
                I += joint_p_distrib[j][i] * math.log(arg, 2)
    return I

def cond_joint_entropy(marginal_p_distrib_Z, joint_p_distrib_XYZ):
    """
    Computes H(X,Y|Z) from P_{X,Y,Z} and P_Z (using the equation H(X,Y|Z) = H(X,Y,Z) - H(Z))
    """
    # H_1 = H(Z)
    H_1 = entropy(marginal_p_distrib_Z)
    # H_2 = H(X,Y,Z)
    H_2 = joint_entropy_3(joint_p_distrib_XYZ)
    # H(X,Y|Z) = H(X,Y,Z) - H(Z)
    H = H_2 - H_1
    return H

def cond_mutual_information(marginal_p_distrib_X, marginal_p_distrib_Z, joint_p_distrib_XZ, joint_p_distrib_YZ, joint_p_distrib_XYZ):
    """
    Computes I(X;Y|Z) from  P_X and P_Z and P_{X,Z} and P_{Y,Z} and P_{X,Y,Z}
    """

    # I_1 = I(X;Z)
    I_1 = mutual_information(marginal_p_distrib_X, marginal_p_distrib_Z, joint_p_distrib_XZ)
    # H_1 = H(X)
    H_1 = entropy(marginal_p_distrib_X)
    # Transpose of joint distribution to be coherent with function joint_cond_entropy
    joint_p_distrib_YXZ = np.transpose(joint_p_distrib_XYZ, (1, 0, 2))
    # H_2 = H(X|Y,Z)
    H_2 = joint_cond_entropy(joint_p_distrib_YXZ, joint_p_distrib_YZ)
    # I(X;Y|Z) = - I(X;Z) + H(X) - H(X|Y,Z)
    I = -I_1 + H_1 - H_2
    return I


def joint_p(param_1_values, param_2_values):
    """Computes joint probability distribuion P(param_1, param_2)"""
    param_1_unique, param_1_counts = np.unique(param_1_values, return_counts=True)
    param_2_unique, param_2_counts = np.unique(param_2_values, return_counts=True)
    n = len(param_1_unique)
    m = len(param_2_unique)
    joint_p_distrib = np.zeros([m,n]) # Lines~param_2, columns~param_1
    for i, mod_1 in enumerate(param_1_unique):
        for j, mod_2 in enumerate(param_2_unique):
            for k in range(len(param_1_values)):
                if param_1_values[k]==mod_1 and param_2_values[k]==mod_2:
                    joint_p_distrib[j][i] += 1/len(param_1_values)
    return joint_p_distrib

def joint_p_3(param_1_values, param_2_values, param_3_values):
    """Computes joint probability distribuion P(param_1, param_2, param_3)"""
    param_1_unique, param_1_counts = np.unique(param_1_values, return_counts=True)
    param_2_unique, param_2_counts = np.unique(param_2_values, return_counts=True)
    param_3_unique, param_3_counts = np.unique(param_3_values, return_counts=True)
    n = len(param_1_unique)
    m = len(param_2_unique)
    p = len(param_3_unique)
    joint_p_distrib = np.zeros([m,n,p]) # Lines~param_2, columns~param_1, third dim~param_3
    for i, mod_1 in enumerate(param_1_unique):
        for j, mod_2 in enumerate(param_2_unique):
            for k, mod_3 in enumerate(param_3_unique):
                for l in range(len(param_1_values)):
                    if param_1_values[l]==mod_1 and param_2_values[l]==mod_2 and param_3_values[l]==mod_3:
                        joint_p_distrib[j][i][k] += 1/len(param_1_values)
    return joint_p_distrib

def conditional_p(param_1_values, param_2_values):
    """Computes conditional probability distribution P(param_1|param_2)"""
    param_1_unique = np.unique(param_1_values)
    param_2_unique, param_2_counts = np.unique(param_2_values, return_counts=True)
    n = len(param_1_unique)
    m = len(param_2_unique)
    conditional_p_distrib = np.zeros([m,n]) # Lines~param_1, columns~param_2
    for i, mod in enumerate(param_2_unique):
        for j, param_val in enumerate(param_2_values):
            if(param_val == mod):
                param_2_idx = np.where(param_1_unique == param_1_values[j])
                conditional_p_distrib[i][param_2_idx] += 1/param_2_counts[i]
    return conditional_p_distrib

def joint_entropy_3(joint_p_distrib):
    """
    Computes H(X,Y,Z) from P_{X,Y,Z}
    """

    H = 0
    n = np.shape(joint_p_distrib)[1]
    m = np.shape(joint_p_distrib)[0]
    p = np.shape(joint_p_distrib)[2]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                arg = joint_p_distrib[j][i][k]
                if(arg !=0):
                    H -= arg*math.log(arg, 2)
    return H

def joint_cond_entropy(joint_p_distrib_XYZ, joint_p_distrib_XZ):
    """
    Computes H(Y|X,Z) from P_{X,Y,Z} and P_{X,Z}
    """

    n = np.shape(joint_p_distrib_XYZ)[1] # X
    m = np.shape(joint_p_distrib_XYZ)[0] # Y
    p = np.shape(joint_p_distrib_XYZ)[2] # Z
    H = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                arg = joint_p_distrib_XYZ[j][i][k]/joint_p_distrib_XZ[k][i]
                if arg != 0:
                    H -= joint_p_distrib_XYZ[j][i][k] * math.log(arg, 2)
    return H

if __name__ == "__main__":

    # -- MEDICAL DIAGNOSIS -- #

    print("\n---Medical diagnosis---")
    medicalDF = pd.read_csv("P1_medicalDB.csv", delimiter=',', header=0)

    # -- 6 -- #

    star = '*' * 14
    print("\n" + star)
    print("* QUESTION 6 *")
    print(star)
    print("\n{:<10}{:^30}{:>10}".format("Parameter", "Entropy (in Shannon)", "Cardinality"))
    dash = '-' * 60
    print("\n{}".format(dash))
    entropies = []
    marginal_probas = []
    for param in medicalDF:
        unique, counts = np.unique(medicalDF[param].values, return_counts=True)
        proba = []
        for i, mod in enumerate(unique):
            proba.append(counts[i]/sum(counts))
        h = entropy(proba)
        print("{:<10}{:^30}{:>10}".format(param, h, len(unique)))
        print(dash)
        marginal_probas.append(proba)
        entropies.append(h)
    marginal_probas = dict(zip(medicalDF.columns, marginal_probas))

    # -- 7 -- #

    print("\n{:^50}".format("*"))
    print("\n{:^50}".format("*     *"))
    print("\n" + star)
    print("* QUESTION 7 *")
    print(star)
    print("\n{:<10}{:^40}".format("Parameter", "Conditional entropy of the disease (in Shannon)"))
    dash = '-' * 60
    print("\n{}".format(dash))
    DIS_values = medicalDF["DIS"].values
    for param in medicalDF:
        if param == "DIS":
            continue
        param_values = medicalDF[param].values
        conditional_p_distrib = conditional_p(DIS_values, param_values)
        cond_h = conditional_entropy(conditional_p_distrib, marginal_probas[param])
        if param == "age":
            conditional_H_age= cond_h # term used for question 11
        print("{:<10}{:^40}".format(param, cond_h))
        print(dash)

    # -- 8 -- #

    print("\n{:^50}".format("*"))
    print("\n{:^50}".format("*     *"))
    print("\n" + star)
    print("* QUESTION 8 *")
    print(star)
    obesity_values = medicalDF["obesity"].values
    age_values = medicalDF["age"].values
    I_obesity_age = mutual_information(marginal_probas["obesity"], marginal_probas["age"], joint_p(obesity_values, age_values))
    print("\nI(obesity; age) = ", I_obesity_age)

    # -- 9 -- #

    print("\n{:^50}".format("*"))
    print("\n{:^50}".format("*     *"))
    print("\n" + star)
    print("* QUESTION 9 *")
    print(star)
    print("\n"+dash)
    I_DIS = []
    for param in medicalDF:
        if param == "DIS":
            continue
        param_values = medicalDF[param].values
        I_DIS_param = mutual_information(marginal_probas["DIS"], marginal_probas[param], joint_p(DIS_values, param_values))
        I_DIS.append(I_DIS_param.copy())
        print("I(DIS; {}) = {}".format(param, I_DIS_param))
        print(dash)
    var = list(medicalDF.columns.values)
    var.remove("DIS")
    print("\nThe mutual information is maximized for variable {}".format(var[I_DIS.index(max(I_DIS))]))

    # -- 10 -- #

    print("\n{:^50}".format("*"))
    print("\n{:^50}".format("*     *"))
    print("\n" + star)
    print("* QUESTION 10 *")
    print(star)
    print("\n"+dash)
    medicalDF_10 = medicalDF.copy()
    medicalDF_10 = medicalDF_10.drop(medicalDF_10[medicalDF.DIS == "PBC"].index)
    DIS_values_10 = medicalDF_10["DIS"].values

    marginal_probas_10 = []
    for param in medicalDF_10:
        unique, counts = np.unique(medicalDF_10[param].values, return_counts=True)
        proba_10 = []
        for i, mod in enumerate(unique):
            proba_10.append(counts[i]/sum(counts))
        h_10 = entropy(proba_10)
        marginal_probas_10.append(proba_10)
    marginal_probas_10 = dict(zip(medicalDF_10.columns, marginal_probas_10))

    marginal_probas_DIS = marginal_probas_10["DIS"]

    I_DIS = []
    for param in medicalDF_10:
        if param == "DIS":
            continue
        param_values = medicalDF_10[param].values
        I_DIS_param = mutual_information(marginal_probas_DIS, marginal_probas_10[param], joint_p(DIS_values_10, param_values))
        I_DIS.append(I_DIS_param)
        print("I(DIS; {}) = {}".format(param, I_DIS_param))
        print(dash)
    var = list(medicalDF_10.columns.values)
    var.remove("DIS")
    print("\nThe mutual information is maximized for variable {}".format(var[I_DIS.index(max(I_DIS))]))

    print("\n"+star)
    print("\n{:<10}{:^40}".format("Parameter", "Conditional entropy of the disease (in Shannon)"))
    dash = '-' * 60
    print("\n{}".format(dash))
    for param in medicalDF_10:
        if param == "DIS":
            continue
        param_values = medicalDF_10[param].values
        conditional_p_distrib = conditional_p(DIS_values_10, param_values)
        cond_h = conditional_entropy(conditional_p_distrib, marginal_probas_10[param])
        print("{:<10}{:^40}".format(param, cond_h))
        print(dash)

    # -- 11 -- #

    print("\n{:^50}".format("*"))
    print("\n{:^50}".format("*     *"))
    print("\n" + star)
    print("* QUESTION 11 *")
    print(star)
    print("\n"+dash)
    p_dis_age = joint_p(DIS_values, age_values)

    I_DIS_age_known = []
    for param in medicalDF:
        if param == "DIS" or param == "age":
            continue
        param_values = medicalDF[param].values
        I_DIS_param_age_known = cond_mutual_information(marginal_probas["DIS"], marginal_probas["age"], p_dis_age, joint_p(param_values, age_values), joint_p_3(DIS_values, param_values, age_values))
        I_DIS_age_known.append(I_DIS_param_age_known.copy())
        print("I(DIS; {}| age) = {}".format(param, I_DIS_param_age_known))
        print(dash)
    var = list(medicalDF.columns.values)
    var.remove("DIS")
    var.remove("age")
    print("\nThe mutual information, given the age, is maximized for variable {}".format(var[I_DIS_age_known.index(max(I_DIS_age_known))]))

    print("\n"+star)
    print("\nH(DIS|age) = {}\n".format(conditional_H_age))
    I_DIS_age_known = np.asarray(I_DIS_age_known)
    conditional_H_age_X = conditional_H_age - I_DIS_age_known
    i = 0
    for param in medicalDF:
        if param == "DIS" or param == "age":
            continue
        print("H(DIS| {}, age) = {}".format(param, conditional_H_age_X[i]))
        print(dash)
        i += 1
