
# coding: utf-8

# version 1.0.2
# #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# # **Introduction to Machine Learning with Apache Spark**
# ## **Predicting Movie Ratings**
# #### One of the most common uses of big data is to predict what users want.  This allows Google to show you relevant ads, Amazon to recommend relevant products, and Netflix to recommend movies that you might like.  This lab will demonstrate how we can use Apache Spark to recommend movies to a user.  We will start with some basic techniques, and then use the [Spark MLlib][mllib] library's Alternating Least Squares method to make more sophisticated predictions.
# #### For this lab, we will use a subset dataset of 500,000 ratings we have included for you into your VM (and on Databricks) from the [movielens 10M stable benchmark rating dataset](http://grouplens.org/datasets/movielens/). However, the same code you write will work for the full dataset, or their latest dataset of 21 million ratings.
# #### In this lab:
# #### *Part 0*: Preliminaries
# #### *Part 1*: Basic Recommendations
# #### *Part 2*: Collaborative Filtering
# #### *Part 3*: Predictions for Yourself
# #### As mentioned during the first Learning Spark lab, think carefully before calling `collect()` on any datasets.  When you are using a small dataset, calling `collect()` and then using Python to get a sense for the data locally (in the driver program) will work fine, but this will not work when you are using a large dataset that doesn't fit in memory on one machine.  Solutions that call `collect()` and do local analysis that could have been done with Spark will likely fail in the autograder and not receive full credit.
# [mllib]: https://spark.apache.org/mllib/

# ### Code
# #### This assignment can be completed using basic Python and pySpark Transformations and Actions.  Libraries other than math are not necessary. With the exception of the ML functions that we introduce in this assignment, you should be able to complete all parts of this homework using only the Spark functions you have used in prior lab exercises (although you are welcome to use more features of Spark if you like!).

# In[1]:

import sys
import os
from test_helper import Test

baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab4', 'small')

ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')


# ### **Part 0: Preliminaries**
# #### We read in each of the files and create an RDD consisting of parsed lines.
# #### Each line in the ratings dataset (`ratings.dat.gz`) is formatted as:
# ####   `UserID::MovieID::Rating::Timestamp`
# #### Each line in the movies (`movies.dat`) dataset is formatted as:
# ####   `MovieID::Title::Genres`
# #### The `Genres` field has the format
# ####   `Genres1|Genres2|Genres3|...`
# #### The format of these files is uniform and simple, so we can use Python [`split()`](https://docs.python.org/2/library/stdtypes.html#str.split) to parse their lines.
# #### Parsing the two files yields two RDDS
# * #### For each line in the ratings dataset, we create a tuple of (UserID, MovieID, Rating). We drop the timestamp because we do not need it for this exercise.
# * #### For each line in the movies dataset, we create a tuple of (MovieID, Title). We drop the Genres because we do not need them for this exercise.

# In[2]:

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)

def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]


ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print 'There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount)
print 'Ratings: %s' % ratingsRDD.take(3)
print 'Movies: %s' % moviesRDD.take(3)

assert ratingsCount == 487650
assert moviesCount == 3883
assert moviesRDD.filter(lambda (id, title): title == 'Toy Story (1995)').count() == 1
assert (ratingsRDD.takeOrdered(1, key=lambda (user, movie, rating): movie)
        == [(1, 1, 5.0)])


# #### In this lab we will be examining subsets of the tuples we create (e.g., the top rated movies by users). Whenever we examine only a subset of a large dataset, there is the potential that the result will depend on the order we perform operations, such as joins, or how the data is partitioned across the workers. What we want to guarantee is that we always see the same results for a subset, independent of how we manipulate or store the data.
# #### We can do that by sorting before we examine a subset. You might think that the most obvious choice when dealing with an RDD of tuples would be to use the [`sortByKey()` method][sortbykey]. However this choice is problematic, as we can still end up with different results if the key is not unique.
# #### Note: It is important to use the [`unicode` type](https://docs.python.org/2/howto/unicode.html#the-unicode-type) instead of the `string` type as the titles are in unicode characters.
# #### Consider the following example, and note that while the sets are equal, the printed lists are usually in different order by value, *although they may randomly match up from time to time.*
# #### You can try running this multiple times.  If the last assertion fails, don't worry about it: that was just the luck of the draw.  And note that in some environments the results may be more deterministic.
# [sortbykey]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sortByKey

# In[4]:

tmp1 = [(1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'delta')]
tmp2 = [(1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'alpha')]

oneRDD = sc.parallelize(tmp1)
twoRDD = sc.parallelize(tmp2)
oneSorted = oneRDD.sortByKey(True).collect()
twoSorted = twoRDD.sortByKey(True).collect()
print oneSorted
print twoSorted
assert set(oneSorted) == set(twoSorted)     # Note that both lists have the same elements
assert twoSorted[0][0] < twoSorted.pop()[0] # Check that it is sorted by the keys
assert oneSorted[0:2] != twoSorted[0:2]     # Note that the subset consisting of the first two elements does not match


# #### Even though the two lists contain identical tuples, the difference in ordering *sometimes* yields a different ordering for the sorted RDD (try running the cell repeatedly and see if the results change or the assertion fails). If we only examined the first two elements of the RDD (e.g., using `take(2)`), then we would observe different answers - **that is a really bad outcome as we want identical input data to always yield identical output**. A better technique is to sort the RDD by *both the key and value*, which we can do by combining the key and value into a single string and then sorting on that string. Since the key is an integer and the value is a unicode string, we can use a function to combine them into a single unicode string (e.g., `unicode('%.3f' % key) + ' ' + value`) before sorting the RDD using [sortBy()][sortby].
# [sortby]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sortBy

# In[5]:

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)


print oneRDD.sortBy(sortFunction, True).collect()
print twoRDD.sortBy(sortFunction, True).collect()


# #### If we just want to look at the first few elements of the RDD in sorted order, we can use the [takeOrdered][takeordered] method with the `sortFunction` we defined.
# [takeordered]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.takeOrdered

# In[6]:

oneSorted1 = oneRDD.takeOrdered(oneRDD.count(),key=sortFunction)
twoSorted1 = twoRDD.takeOrdered(twoRDD.count(),key=sortFunction)
print 'one is %s' % oneSorted1
print 'two is %s' % twoSorted1
assert oneSorted1 == twoSorted1


# ### **Part 1: Basic Recommendations**
# #### One way to recommend movies is to always recommend the movies with the highest average rating. In this part, we will use Spark to find the name, number of ratings, and the average rating of the 20 movies with the highest average rating and more than 500 reviews. We want to filter our movies with high ratings but fewer than or equal to 500 reviews because movies with few reviews may not have broad appeal to everyone.

# #### **(1a) Number of Ratings and Average Ratings for a Movie**
# #### Using only Python, implement a helper function `getCountsAndAverages()` that takes a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...)) and returns a tuple of (MovieID, (number of ratings, averageRating)). For example, given the tuple `(100, (10.0, 20.0, 30.0))`, your function should return `(100, (3, 20.0))`

# In[9]:

# First, implement a helper function `getCountsAndAverages` using only Python
def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    movieId = IDandRatingsTuple[0]
    ratings = IDandRatingsTuple[1]
    return (movieId, (len(ratings), float(sum(ratings)) / len(ratings)))


# In[10]:

# TEST Number of Ratings and Average Ratings for a Movie (1a)

Test.assertEquals(getCountsAndAverages((1, (1, 2, 3, 4))), (1, (4, 2.5)),
                            'incorrect getCountsAndAverages() with integer list')
Test.assertEquals(getCountsAndAverages((100, (10.0, 20.0, 30.0))), (100, (3, 20.0)),
                            'incorrect getCountsAndAverages() with float list')
Test.assertEquals(getCountsAndAverages((110, xrange(20))), (110, (20, 9.5)),
                            'incorrect getCountsAndAverages() with xrange')


# #### **(1b) Movies with Highest Average Ratings**
# #### Now that we have a way to calculate the average ratings, we will use the `getCountsAndAverages()` helper function with Spark to determine movies with highest average ratings.
# #### The steps you should perform are:
# * #### Recall that the `ratingsRDD` contains tuples of the form (UserID, MovieID, Rating). From `ratingsRDD` create an RDD with tuples of the form (MovieID, Python iterable of Ratings for that MovieID). This transformation will yield an RDD of the form: `[(1, <pyspark.resultiterable.ResultIterable object at 0x7f16d50e7c90>), (2, <pyspark.resultiterable.ResultIterable object at 0x7f16d50e79d0>), (3, <pyspark.resultiterable.ResultIterable object at 0x7f16d50e7610>)]`. Note that you will only need to perform two Spark transformations to do this step.
# * #### Using `movieIDsWithRatingsRDD` and your `getCountsAndAverages()` helper function, compute the number of ratings and average rating for each movie to yield tuples of the form (MovieID, (number of ratings, average rating)). This transformation will yield an RDD of the form: `[(1, (993, 4.145015105740181)), (2, (332, 3.174698795180723)), (3, (299, 3.0468227424749164))]`. You can do this step with one Spark transformation
# * #### We want to see movie names, instead of movie IDs. To `moviesRDD`, apply RDD transformations that use `movieIDsWithAvgRatingsRDD` to get the movie names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form (average rating, movie name, number of ratings). This set of transformations will yield an RDD of the form: `[(1.0, u'Autopsy (Macchie Solari) (1975)', 1), (1.0, u'Better Living (1998)', 1), (1.0, u'Big Squeeze, The (1996)', 3)]`. You will need to do two Spark transformations to complete this step: first use the `moviesRDD` with `movieIDsWithAvgRatingsRDD` to create a new RDD with Movie names matched to Movie IDs, then convert that RDD into the form of (average rating, movie name, number of ratings). These transformations will yield an RDD that looks like: `[(3.6818181818181817, u'Happiest Millionaire, The (1967)', 22), (3.0468227424749164, u'Grumpier Old Men (1995)', 299), (2.882978723404255, u'Hocus Pocus (1993)', 94)]`

# In[24]:

# From ratingsRDD with tuples of (UserID, MovieID, Rating) create an RDD with tuples of
# the (MovieID, iterable of Ratings for that MovieID)
movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda (userID, movieID, rating): (movieID, [rating]))
                          .reduceByKey(lambda ratingList1, ratingList2: ratingList1 + ratingList2))
print 'movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3)

# Using `movieIDsWithRatingsRDD`, compute the number of ratings and average rating for each movie to
# yield tuples of the form (MovieID, (number of ratings, average rating))
movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)
print 'movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3)

# To `movieIDsWithAvgRatingsRDD`, apply RDD transformations that use `moviesRDD` to get the movie
# names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form
# (average rating, movie name, number of ratings)
movieNameWithAvgRatingsRDD = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda (movieID, titleAndRatingsTuple): (titleAndRatingsTuple[1][1], titleAndRatingsTuple[0], titleAndRatingsTuple[1][0])))
print 'movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3)


# In[25]:

# TEST Movies with Highest Average Ratings (1b)

Test.assertEquals(movieIDsWithRatingsRDD.count(), 3615,
                'incorrect movieIDsWithRatingsRDD.count() (expected 3615)')
movieIDsWithRatingsTakeOrdered = movieIDsWithRatingsRDD.takeOrdered(3)
Test.assertTrue(movieIDsWithRatingsTakeOrdered[0][0] == 1 and
                len(list(movieIDsWithRatingsTakeOrdered[0][1])) == 993,
                'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[0] (expected 993)')
Test.assertTrue(movieIDsWithRatingsTakeOrdered[1][0] == 2 and
                len(list(movieIDsWithRatingsTakeOrdered[1][1])) == 332,
                'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[1] (expected 332)')
Test.assertTrue(movieIDsWithRatingsTakeOrdered[2][0] == 3 and
                len(list(movieIDsWithRatingsTakeOrdered[2][1])) == 299,
                'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[2] (expected 299)')

Test.assertEquals(movieIDsWithAvgRatingsRDD.count(), 3615,
                'incorrect movieIDsWithAvgRatingsRDD.count() (expected 3615)')
Test.assertEquals(movieIDsWithAvgRatingsRDD.takeOrdered(3),
                [(1, (993, 4.145015105740181)), (2, (332, 3.174698795180723)),
                 (3, (299, 3.0468227424749164))],
                'incorrect movieIDsWithAvgRatingsRDD.takeOrdered(3)')

Test.assertEquals(movieNameWithAvgRatingsRDD.count(), 3615,
                'incorrect movieNameWithAvgRatingsRDD.count() (expected 3615)')
Test.assertEquals(movieNameWithAvgRatingsRDD.takeOrdered(3),
                [(1.0, u'Autopsy (Macchie Solari) (1975)', 1), (1.0, u'Better Living (1998)', 1),
                 (1.0, u'Big Squeeze, The (1996)', 3)],
                 'incorrect movieNameWithAvgRatingsRDD.takeOrdered(3)')
