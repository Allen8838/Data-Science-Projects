The input data file is organized as follows

+--------------+---------------------------------------------+--------------------+-------------------------------+-------+-------+
| time         | message_id                                  | sender             | recipient                     | topic | mode  |
+--------------+---------------------------------------------+--------------------+-------------------------------+-------+-------+
| 896308260000 | <GKNWYZTKAVMKBCTMA3ZZSQIBITG5OD1KA@zlsvr22> | Christopher Behney | Toni P Schulenburg|mary hain  |       | email |
+--------------+---------------------------------------------+--------------------+-------------------------------+-------+-------+
| 909790980000 | <N5QXQ4CHS04FBSJO3TALWRBXATDTIWZ0B@zlsvr22> | mark legal taylor  | Marc.R.Cutler@BankAmerica.com |       | email |
+--------------+---------------------------------------------+--------------------+-------------------------------+-------+-------+

there are approximately 200k plus rows of data. The topic column is always blank and mode is always email. Note that the headers were not in
the original data file but displayed here for clarity. 

The challenge for this exam is to write a script that will create one csv file and two visualizations. For the first question, the csv file
that needs to be created is a table with three columns: person, sent, and received. The sent and received columns are counts of how many
emails were sent or received by a given person. The table will be sorted in descending order, based on the number of sent emails. 


First part of my script involves adding headers to the file as well as cleaning the data. The cleaning is from a file called
Dictionary to clean names.csv. Some of the sender's name in the original data file had multiple names for one person, e.g.
mark legal taylor and mark taylor are the same person. The Dictionary to clean names.csv file has a column for the original names as presented
in the original dataset and a column for the revised name that I will be using. Thus, from the above example, every instance of 
mark legal taylor or variations of this name will be transformed to mark taylor. This will allow us to accurately group people's name and
get a correct count of emails sent. The same process has not been carried out in this script for the recipient's column but will be added
later on. 


