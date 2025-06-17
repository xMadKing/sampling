## Method Overview

The initial method involved conducting a manual visual analysis of the distributions, which are stored under the `plots` directory. Through this analysis, it was observed that the distribution appeared to be normally distributed. Among the categories, **B** was the most common, followed by **C**, **A**, and lastly **E** and **D**.

The sampling method analyzes the dataset to compute the mean, standard deviation, and probability for each point category. Using these statistics, new data samples are generated to reflect the observed distributions and category probabilities.

# Verifying 
To verify the similarity between the generated data and the original dataset, I conducted a T-test. The results indicated no significant difference between the two, supporting the validity of the sampling method. Additionally, the automatic analysis of the data assures we have a good approximation of the mean value, standard deviation, and probability for each category, further indicicating that the new data is similar to the original. 

# Approach is generalizable 
My approach works to generate similar data regardless whether the original code is available or not. So long as the data is formatted corrected (I.e same format as current data) my approach will work to generated similar data without previously knowing the parameters. 

# Areas for potential improvement.
Since I knew the data is normally distributed form the code analysis as well as the visual analyis, I could just use a T-test to see if there is any significant difference between the data I generate and the original data. This works as long as the assumption that the original data is normally distributed can be made. I can potentially improve this by avoiding this assumption and conducting a test to verify whether the distribution is normal or not (Possibly a shapiro-wilk U test)