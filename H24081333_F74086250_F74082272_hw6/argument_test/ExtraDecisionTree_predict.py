import DataPreprocess as DP
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score
from IPython.display import clear_output #for jupyter display

X_train, y_train, df_test = DP.func()

X_train_save = X_train
y_train_save = y_train

self_testing = False
optimal = False
Optimal_Var1,Optimal_Var2 = 0,0

def test(Var_name = "show_all",Srand_s = 29,Srand_e = 29,Tree_num_s = 30,Tree_num_e = 30,Srand2 = 2,Min_Split=7,Min_Leaf=2,Max_depth = None):
    global self_testing,optimal
    finalscore = 0
    Var1 = 0
    Var2 = 0
    display_temp = [0,0]
    if optimal:
        Srand_s,Srand_e = Optimal_Var1,Optimal_Var1
        Tree_num_s,Tree_num_e = Optimal_Var2,Optimal_Var2
    for i in range(Srand_s,Srand_e+1):
        for j in range(Tree_num_s,Tree_num_e+1):
            X_train, y_train= X_train_save, y_train_save
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = i, stratify=y_train)
    
            classifier =  ExtraTreesClassifier(criterion='gini',n_estimators=j, max_depth=Max_depth, min_samples_split=Min_Split,min_samples_leaf=Min_Leaf, random_state=Srand2)
            classifier.fit(X_train, y_train)

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train_stdnorm = scaler.transform(X_train)
            classifier.fit(X_train_stdnorm, y_train)
            y_predict = classifier.predict(X_test)
            X_test_stdnorm = scaler.transform(X_test)
            y_predict = classifier.predict(X_test_stdnorm)

            acc = accuracy_score(y_predict, y_test)
            prec = precision_score(y_predict, y_test)
            f1score = f1_score(y_predict, y_test)

            if self_testing:
                if (acc*0.3+prec*0.4+f1score*0.3) > finalscore:
                    finalscore = acc*0.3+prec*0.4+f1score*0.3
                    Var1,Var2 = i,j
                    clear_output()
                    print("Progress ",display_temp[0],"% | ","\u25B0"*display_temp[1])
                    print("Srand = ",i,"\nTree_num = ",j,"\n"+"\u25B0"*20,"\nAccuracy : ",acc,"\nPrecision : ",prec,"\nF1score : ",f1score,"\nScore : ",acc*0.3+prec*0.4+f1score*0.3)
                    print("Current final(max) = ",finalscore,"\nSrand(record) = ",Var1,"\nTree_num(record) = ",Var2)
                    print()

                Total = (Srand_e-Srand_s+1)*(Tree_num_e-Tree_num_s+1)
                progress =  (i-Srand_s)*(Tree_num_e-Tree_num_s+1)+(j-Tree_num_s)
                if int((progress/Total)*100) != display_temp[0] or int((progress/Total)*50) != display_temp[1]:
                    display_temp = [round((progress/Total)*100,2),int((progress/Total)*50)]
                    clear_output()
                    print("Progress ",display_temp[0],"% | ","\u25B0"*display_temp[1])
                    print("Srand = ",i,"\nTree_num = ",j,"\n"+"\u25B0"*20,"\nAccuracy : ",acc,"\nPrecision : ",prec,"\nF1score : ",f1score,"\nScore : ",acc*0.3+prec*0.4+f1score*0.3)
                    print("Current final(max) = ",finalscore,"\nSrand(record) = ",Var1,"\nTree_num(record) = ",Var2)
            
            if Var_name == "Score" and not self_testing:
                print("Srand = ",i,"\nTrees = ",j,"\nMax_depth = ",Max_depth,"\nMin_samples_split = ",Min_Split,"\nMin_leaf = ",Min_Leaf,"\n"+"\u25B0"*20,"\nScore : ",acc*0.3+prec*0.4+f1score*0.3)
            if Var_name == "Accuracy" and not self_testing:
                print("Srand = ",i,"\nTrees = ",j,"\nMax_depth = ",Max_depth,"\nMin_samples_split = ",Min_Split,"\nMin_leaf = ",Min_Leaf,"\n"+"\u25B0"*20,"\nAccuracy : ",acc)
            if Var_name == "Precision" and not self_testing:
                print("Srand = ",i,"\nTrees = ",j,"\nMax_depth = ",Max_depth,"\nMin_samples_split = ",Min_Split,"\nMin_leaf = ",Min_Leaf,"\n"+"\u25B0"*20,"\nPrecision : ",prec)
            if Var_name == "F1score" and not self_testing:
                print("Srand = ",i,"\nTrees = ",j,"\nMax_depth = ",Max_depth,"\nMin_samples_split = ",Min_Split,"\nMin_leaf = ",Min_Leaf,"\n"+"\u25B0"*20,"\nF1score : ",f1score)
            if Var_name == "show_all" and not self_testing:
                print("Srand = ",i,"\nTrees = ",j,"\nMax_depth = ",Max_depth,"\nMin_samples_split = ",Min_Split,"\nMin_leaf = ",Min_Leaf,"\n"+"\u25B0"*20,"\nAccuracy : ",acc,"\nPrecision : ",prec,"\nF1score : ",f1score,"\nScore : ",acc*0.3+prec*0.4+f1score*0.3)
            if not self_testing:
                print()    
    if self_testing:
        return Var1,Var2
    return

def self_optimizing(v1=15,v2=30,v3=20,v4=40,Apply = True):
    global self_testing,optimal,Optimal_Var1,Optimal_Var2
    self_testing = True
    if not optimal:
        temp1,temp2 = test("show_all",v1,v2,v3,v4)
        self_testing = False
        clear_output()
        print("Progress 100% | ","\u25B0"*50)
        print(temp1,temp2)
    else:
        print("Already optimized")
        print(Optimal_Var1,Optimal_Var2)
        return
    if Apply:
        Optimal_Var1,Optimal_Var2 = temp1,temp2
        optimal = True
    return 