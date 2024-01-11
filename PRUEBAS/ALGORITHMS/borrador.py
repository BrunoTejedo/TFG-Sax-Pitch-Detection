# 2 maneras de hacer un csv de un array numpy

DF = pd.DataFrame(pitch_values)
# save the dataframe as a csv file
DF.to_csv("prueba2.csv")
np.savetxt('prueba3.csv', pitch_values) # delimiter????