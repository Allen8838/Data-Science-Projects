The input data file is organized as follows

---------------|---------------------------------------------|--------------------|-------------------------------|-------|-------|
| time         | message_id                                  | sender             | recipient                     | topic | mode  |

| 896308260000 | <GKNWYZTKAVMKBCTMA3ZZSQIBITG5OD1KA@zlsvr22> | Christopher Behney | Toni P Schulenburg|mary hain  |       | email |

| 909790980000 | <N5QXQ4CHS04FBSJO3TALWRBXATDTIWZ0B@zlsvr22> | mark legal taylor  | Marc.R.Cutler@BankAmerica.com |       | email |


there are approximately 200k plus rows of data. The topic column is always blank and mode is always email. Note that the headers were not in
the original data file but displayed here for clarity. 

The challenge for this exam is to write a script that will create one csv file and two visualizations. For the first question, the csv file
that needs to be created is a table with three columns: person, sent, and received. The sent and received columns are counts of how many
emails were sent or received by a given person. The table will be sorted in descending order, based on the number of sent emails. 

## Data Preprocessing 
First part of my script involves adding headers to the file as well as cleaning the data. The cleaning is from a file called
Dictionary to clean names.csv. Some of the sender's name in the original data file had multiple names for one person, e.g.
mark legal taylor and mark taylor are the same person. The Dictionary to clean names.csv file has a column for the original names as presented
in the original dataset and a column for the revised name that I will be using. Thus, from the above example, every instance of 
mark legal taylor or variations of this name will be transformed to mark taylor. This will allow us to accurately group people's name and
get a correct count of emails sent. The same process has not been carried out in this script for the recipient's column but will be added
later on. 

I converted all sender's name to lowercase so names would be aligned when grouping. I also converted the unix time to regular time but kept
only the year and the month. I initially included "day" in the time conversion but found that it was incredibly slow to plot a graph with 
time as an axis. This was because the top sender had sent thousands of emails and converting each sent date to a datetime object and appending
to a list for later graphing took the program a long time. By rounding the dates to a month, I would be able to batch the dates together and
passed in batched dates to the datetime object at once, speeding the process. 

## Question 1
I then created a dictionary called dict_for_number_msgs_sent_by_sender. The resulting dictionary would look something like this for example:
{jeff dasovich:1000, sara shackleton: 900....}. This would be used for the sent column for question 1. At the same time, I created a dictionary
called dict_senders_number_msgs_per_time and list_of_senders_time. The dictionary looks like 
{jeff dasovich :[(1, '2002-06')], sara shackleton: [(2, '2002-11')]....} and the list is a list of tuple that looks as follows
[(jeff dasovich , 2002-06), (sara shackleton, 2002-11)...]. The dictionary will be directly used for question two as it gives us the sender's
name, and number of messages that was sent in a given period of time. 

The list is created by going in the original dataset and making a tuple of both the sender and the time they sent a message. This will be useful for question 3, as I will expand the "recipients" column by parsing the
column by pipes and creating a separate dataframe from that one column. I will then go through that dataframe, convert the value of each cell to
"False" if the cell value is not one of the top 5 senders names and record the column and row address where the cell value is one of the
top senders. Thus, a list of tuples for the sender and time is needed as I will need to associate this tuple with the relevant recipient
and this also avoids having to go through the dataframe row by row, which is time expensive. 

 
