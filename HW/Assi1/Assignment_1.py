
# coding: utf-8

#### 1. Explore the general characteristics of the data, by computing the means and standard deviations of the numerical attributes, as well as the the distributions of male and female customers, the preferred movie genres, etc.

# In[1]:

import numpy as np
vstable = np.genfromtxt("Video_Store.csv", delimiter=",", dtype=None)
labels = vstable[0]
#print vstable
vstable = vstable[1:]


# In[2]:

income = np.array(vstable[:,2], dtype=int)
age = np.array(vstable[:,3], dtype=int)
rentals = np.array(vstable[:,4], dtype=int)
visit_avg = np.array(vstable[:,5], dtype=float)
print "       Income \t\tAge \t\tRentals \tAvg Per Visit"
print "Mean: ", income.mean(), "\t\t", age.mean(), "\t\t", rentals.mean(), "\t\t", visit_avg.mean()
print "Std:  ", income.std(), "\t", age.std(), "\t", rentals.std(), "\t", visit_avg.std()


# In[3]:

table_f = vstable[vstable[:,1] == 'F']
income_f = np.array(table_f[:,2], dtype=int)
age_f = np.array(table_f[:,3], dtype=int)
rentals_f = np.array(table_f[:,4], dtype=int)
visit_avg_f = np.array(table_f[:,5], dtype=float)
print "              Income \t\tAge \t\tRentals \tAvg Per Visit"
print "Female Mean: ", income_f.mean(), "\t", age_f.mean(), "\t\t", rentals_f.mean(), "\t\t", visit_avg_f.mean()
print "Female Std:  ", income_f.std(), "\t", age_f.std(), "\t", rentals_f.std(), "\t", visit_avg_f.std()


# In[4]:

action_f = table_f[table_f[:,7] == 'Action']
comedy_f = table_f[table_f[:,7] == 'Comedy']
drama_f = table_f[table_f[:,7] == 'Drama']
print "\t\t  Action  Comedy  Drama"
print "Female Preferred:", len(action_f), "\t ", len(comedy_f), "\t ", len(drama_f)
print "Buy incidentals: ", len(action_f[action_f[:,6] == 'Yes']), "\t ", len(comedy_f[comedy_f[:,6] == 'Yes']), "\t ", len(drama_f[drama_f[:,6] == 'Yes'])
print "Female customer:   ", len(table_f)
print "Female Incidentals:", len(vstable[table_f[:,6] == 'Yes'])


# In[5]:

table_m = vstable[vstable[:,1] == 'M']
income_m = np.array(table_m[:,2], dtype=int)
age_m = np.array(table_m[:,3], dtype=int)
rentals_m = np.array(table_m[:,4], dtype=int)
visit_avg_m = np.array(table_m[:,5], dtype=float)
print "              Income \t\tAge \t\tRentals \tAvg Per Visit"
print "Male Mean: ", income_m.mean(), "\t", age_m.mean(), "\t\t", rentals_m.mean(), "\t", visit_avg_m.mean()
print "Male Std:  ", income_m.std(), "\t", age_m.std(), "\t", rentals_m.std(), "\t", visit_avg_m.std()


# In[6]:

action_m = table_m[table_m[:,7] == 'Action']
comedy_m = table_m[table_m[:,7] == 'Comedy']
drama_m = table_m[table_m[:,7] == 'Drama']
print "\t\t Action Comedy  Drama"
print "Male preferred: ", len(action_m), "\t", len(comedy_m), "\t", len(drama_m)
print "Buy incidentals:", len(action_m[action_m[:,6] == 'Yes']), "\t", len(comedy_m[comedy_m[:,6] == 'Yes']), "\t", len(drama_m[drama_m[:,6] == 'Yes'])
print "Male customer:   ", len(table_m)
print "Male Incidentals:", len(vstable[table_m[:,6] == 'Yes'])


#### 2. Suppose that because of the higher profit margin, the store would like to increase the sales of incidentals. Select the subset of customers who tend to buy incidentals. Then, compute summaries (as in part 1) of the selected data with respect to all other attributes. Can you observe any significant patterns that characterize this segment of customers in contrast to the general customer populations? Based on your observations discuss how this could be accomplished (e.g., should customers with specific characteristics be targeted? Should certain types of movies be preferred? Etc.). Explain your answer based on your analysis of the data.

# Ans: 100% of female and 69% of male who prefer action movie buy incidentals, but 83% of customers who prefer comedy movie don't buy incidentals. Last, 50% of customers who prefer drama movie buy incidentals. Indeed, customer who prefer action movie could be the target especially female in this group.

# In[7]:

buyIncidental = vstable[vstable[:,6]=='Yes']
buyInc_Action = buyIncidental[buyIncidental[:,7] == 'Action']
buyInc_Comedy = buyIncidental[buyIncidental[:,7] == 'Comedy']
buyInc_Drama = buyIncidental[buyIncidental[:,7] == 'Drama']
print "Incidentals bought with movie"
print "Action:", len(buyInc_Action) 
print "Comedy:", len(buyInc_Comedy)
print "Drama: ", len(buyInc_Drama)
print "Total: ", len(buyIncidental)


# In[8]:

incomeInc_f = np.array(action_f[:,2], dtype=int)
ageInc_f = np.array(action_f[:,3], dtype=int)
rentalsInc_f = np.array(action_f[:,4], dtype=int)
visit_avgInc_f = np.array(action_f[:,5], dtype=float)
print "F + A + I:  Income \t\tAge \t\tRentals \tAvg Per Visit"
print "Mean:      ", incomeInc_f.mean(), "\t       ", ageInc_f.mean(), "\t\t", rentalsInc_f.mean(), "\t\t", visit_avgInc_f.mean()
print "Std:       ", incomeInc_f.std(), "\t", ageInc_f.std(), "\t", rentalsInc_f.std(), "\t", visit_avgInc_f.std()


# In[9]:

incomeInc_m = np.array(action_m[:,2], dtype=int)
ageInc_m = np.array(action_m[:,3], dtype=int)
rentalsInc_m = np.array(action_m[:,4], dtype=int)
visit_avgInc_m = np.array(action_m[:,5], dtype=float)
print "M + A + I:  Income \t\tAge \t\tRentals \tAvg Per Visit"
print "Mean:      ", incomeInc_m.mean(), "\t", ageInc_m.mean(), "\t", rentalsInc_m.mean(), "\t", visit_avgInc_m.mean()
print "Std:       ", incomeInc_m.std(), "\t", ageInc_m.std(), "\t", rentalsInc_m.std(), "\t", visit_avgInc_m.std()


#### 3. Use z-score normalization to standardize the values of the Rentals attribute. Show the results side-by-side with the original Rentals attribute. [Do not change the original Rentals attribute in the table.]

# In[10]:

rentals_mean = rentals.mean()
rentals_std = rentals.std()
rentals_znorm = (rentals - rentals_mean) / rentals_std
#rentals_com = np.array([rentals.astype('float'), rentals_znorm]).T
print "Side-by-side Rentals / z-score normalization\n "
for x,y in zip(rentals, rentals_znorm):
    print ("%d, %f\n" % (x,y))


#### 4. Use Min-Max Normalization to transform the values of all numeric attributes (Income, Age, Rentals, Avg. Per Visit) onto the range 0.0-1.0.

# In[11]:

min_income = income.min()
max_income = income.max()
range_income = max_income - min_income
norm_income = (income - min_income).astype(float) / range_income
print "Normalizing Income\n", norm_income


# In[12]:

min_age = age.min()
max_age = age.max()
range_age = max_age - min_age
norm_age = (age - min_age).astype(float) / range_age
print "Normalizing Age\n", norm_age


# In[13]:

min_rentals = rentals.min()
max_rentals = rentals.max()
range_rentals = max_rentals - min_rentals
norm_rentals = (rentals - min_rentals).astype(float) / range_rentals
print "Normalizing Rentals\n", norm_rentals


# In[14]:

min_visit_avg = visit_avg.min()
max_visit_avg = visit_avg.max()
range_visit_avg = max_visit_avg - min_visit_avg
norm_visit_avg = (visit_avg - min_visit_avg) / range_visit_avg
print "Normalizing Visit Average\n", norm_visit_avg


#### 5. Convert the table (after normalization in part 4) into the standard spreadsheet format. Note that this requires converting each categorical attribute into multiple attributes (one for each values of the categorical attribute) and assigning binary values corresponding to the presence or not presence of the attribute value in the original record). For example, the Gender attribute will be transformed into two attributes, "Genre=M" and "Genre=F". The numerical attributes will remain unchanged. This process should result in a new table with 12 attributes (one for Customer ID, two for Gender, one for each of Income, Age, Rentals, Avg. Per Visit, two for Incidentals, and three for Genre). Save this new table into a file called video_store_numeric.csv.

# In[15]:

gender = np.array(vstable[:,1])
gen_f = np.zeros(len(gender))
gen_f[gender=='F'] = 1
#print gen_f
gen_m = np.zeros(len(gender))
gen_m[gender=='M'] = 1
#print gen_m


# In[16]:

incidentals = np.array(vstable[:,6])
inc_y = np.zeros(len(incidentals))
inc_y[incidentals=='Yes'] = 1
#print inc_y
inc_n = np.zeros(len(incidentals))
inc_n[incidentals=='No'] = 1
#print inc_n


# In[17]:

genre = np.array(vstable[:,7])
genre_a = np.zeros(len(genre))
genre_a[genre=='Action'] = 1
#print genre_a
genre_c = np.zeros(len(genre))
genre_c[genre=='Comedy'] = 1
#print genre_c
genre_d = np.zeros(len(genre))
genre_d[genre=='Drama'] = 1
#print genre_d


# In[18]:

custID = np.array(vstable[:,0], dtype=int)
vs_new = np.array([custID, gen_f, gen_m, norm_income, norm_age, norm_rentals, norm_visit_avg, inc_y, inc_n, genre_a, genre_c, genre_d])
vs_new = vs_new.T
#print vs_new
out_file = open("video_store_numeric.csv", "w")
out_file.write("Cust ID,Female,Male,Income,Age,Rentals,Avg Per Visit,Buy Incidentals, Not Buy Incidentals,Action,Comedy,Drama\n" )
np.savetxt(out_file, vs_new, fmt='%d,%d,%d,%1.2f,%1.2f,%1.2f,%1.2f,%d,%d,%d,%d,%d', delimiter=',')
out_file.close()


                6. Using the standardized data set (from part e), perform basic correlation analysis among the attributes. Discuss your results by indicating any strong correlations (positive or negative) among pairs of attributes. You need to construct a complete Correlation Matrix. Be sure to first remove the Customer ID column before creating the correlation matrix. [Hint: you can do this by using the corrcoef function in NumPy].
                
# Ans: Age and income have a strong positive relationship. Age also has a strong positive toward drama movie but strong negative toward action movie.

# In[19]:

norm_table = np.genfromtxt("video_store_numeric.csv", delimiter=",", dtype=None)
norm_table = norm_table[1:]
norm_no_id = norm_table[:,1:].astype(float)
#print norm_no_id
corr_matrix = np.corrcoef(norm_no_id.T)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
print "    F     M     $    Age   Rent AvgVis Inc  NoInc  Act   Com   Dra"
print corr_matrix


#### 7. Using Matplotlib library, create a scatter plot of the (non-normalized) Income attribute relative to Age. Be sure that your plot contains appropriate labels for the axes. Do these variables seem correlated?

# Ans: The variables seem correlated. Low age has low income and older has hier income

# In[20]:

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
ax.scatter(age, income, color="red", marker="*")
ax.set_title("Age vs Income")
ax.set_ylabel("Income")
ax.set_xlabel("Age")
plt.axis([10,80,0,90000])
plt.show()


#### 8. Using the hist function of Matplotlib, create histograms for (non-normalized) Income (using 9 bins) and Age (using 7 bins).

# In[21]:

plt.hist(income, bins=9, alpha=0.5)
plt.xlabel('Income')
plt.ylabel('Count')
plt.title('Histogram of Income Populations')
plt.axis([0,90000,0,10])
plt.grid(True)
plt.show()


# In[22]:

plt.hist(age, bins=7, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age Populations')
plt.axis([10,75,0,15])
plt.grid(True)
plt.show()


#### 9. Using Python and Numpy, perform a cross-tabulation of the two "gender" variables versus the three "genre" variables. This requires the aggregation of the occurrences of each genre separately for each gender. You can use whatever appropriate data structure you which to store the results, but you can display it as as a 2 x 3 (gender x genre) table with entries representing the counts. Then, use Matplotlib to create a bar chart graph to visualize of the relationships between these sets of variables (comparing Male and Female customer across the three Genres). Your chart should contain appropriate labels for axes. [Hint: This example of creating simple bar charts using Matplotlib may be useful.]

# In[23]:

genre_f = np.array([len(action_f), len(comedy_f), len(drama_f)])
genre_m = np.array([len(action_m), len(comedy_m), len(drama_m)])
print "        Action\tComedy\tDrama"
print "Female:  ",genre_f[0], "\t ", genre_f[1], "\t ", genre_f[2]
print "Male:    ", genre_m[0], "\t ", genre_m[1], "\t ", genre_m[2]


# In[24]:

N = 3
ind = np.arange(N)
width = 0.2
fig, ax = plt.subplots()

rects1 = ax.bar(ind, genre_f, width, color='r')
rects2 = ax.bar(ind+width, genre_m, width, color='b')

ax.set_xlabel('Genre')
ax.set_ylabel('Scores')
ax.set_title('Scores by genre and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Action', 'Comedy', 'Drama') )
ax.legend( (rects1[0], rects2[0]), ('Female', 'Male') )
          
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.axis([0,3,0,16])
plt.show()

