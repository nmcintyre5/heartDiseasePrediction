from pyspark.sql import SparkSession
from structs import heart_schema
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.getOrCreate()

# Create dataframe heart_df
heart_df = spark.read.csv('heart_data.csv', header = True, schema = heart_schema)

# Optional: Print the schema
#heart_df.printSchema()

#Optional: Display the dataframe
#heart_df.show()

#Optional: Show summary of descriptive statistics
#heart_df.select("biking", "smoking", "heart_disease").summary().show()

# Create an SQL table in memory
heart_df.createOrReplaceTempView('HEART_DATA')

heart_df = heart_df.drop('index')

# Create an empty dictionary to store correlations
correlation_dict = {}

# Correlation Analysis
print("Correlation to Heart Disease: \n")
for col in heart_df.columns:
    correlation = heart_df.stat.corr('heart_disease', col)
    print('{} = {} '.format(col, correlation))
    correlation_dict[col] = correlation

# Create a new dataframe by capturing it using SQL
df = spark.sql('SELECT biking, smoking FROM HEART_DATA')
# df.printSchema()

# Caputure IV column names of interest
inputlist = df.columns
# print("inputlist: ",inputlist)

#Create a vector with biking & smoking as "features"
features_assembler = VectorAssembler(inputCols=inputlist, outputCol='features')

#Transform the heart_df with df
df = features_assembler.transform(heart_df)
# df.printSchema()

#Create working dataframe that selects the IVs as features and the DV heart_disease
working_df = df.select('features','heart_disease')
#working_df.show()

#Test the heart disease ML modle (70% train, 30% test)
training, test = working_df.randomSplit([0.7,0.3])
#print("\ntest schema:\n")
#test.printSchema()

#Create linear regression object
lr = LinearRegression(featuresCol='features',labelCol='heart_disease')

# Creat model that performs a linear regression fit to the training data
model = lr.fit(training)

# Get coefficients & intercept
print("\nModel coefficients: ",model.coefficients)
print('\nModel intercept: ',model.intercept)

# Get summary of the model
summary = model.summary
print('\nRoot Mean Squared Error (RMSE):', summary.rootMeanSquaredError)
print('\nR-squared (r2) score:', summary.r2)

#Create a dataframe, predict_output
predictions = model.transform(test)
predictions.show()

#Evaluate the model
evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='heart_disease',metricName='r2')
print('r2 of the test data:', evaluator.evaluate(predictions))





