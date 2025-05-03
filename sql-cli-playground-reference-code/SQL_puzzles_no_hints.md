# SQL Puzzles for sql-cli-playground

Here are some SQL puzzles to practice your skills within the `sql-cli-playground` environment. For each puzzle, first run the provided Python code snippets in the CLI to set up the necessary tables. Then, write and execute the SQL query to solve the puzzle.

Give me some high-level approaches for how to solve this SQL puzzle (just the barest hint - DO NOT solve it for me).

---

## Puzzle 1: Simple Selection (Easy)

**Task:** Select the `name` and `species` of all pets from the `pets` table.

**Setup Code (Run in CLI):**

```python
# Run these lines one by one in the CLI
/create pets2
pets2['id'] = [1, 2, 3, 4, 5]
pets2['name'] = ['Buddy', 'Lucy', 'Max', 'Daisy', 'Charlie']
pets2['species'] = ['Dog', 'Cat', 'Dog', 'Dog', 'Cat']
pets2['age'] = [3, 5, 2, 7, 1]
```

**Expected Output:** A table showing only the `name` and `species` columns for all 5 pets.

---

## Puzzle 2: Filtering Data (Easy)

**Task:** Find the names of all dogs older than 3 years from the `pets` table.

**Setup Code:** (Use the `pets` table created in Puzzle 1)

**Expected Output:** A table showing the names 'Daisy'.

---

## Puzzle 3: Aggregation and Grouping (Medium)

**Task:** Calculate the average age for each pet species in the `pets` table.

**Setup Code:** (Use the `pets` table created in Puzzle 1)

**Expected Output:** A table showing each `species` and its corresponding average age (e.g., Dog: 4.0, Cat: 3.0).

---

## Puzzle 4: Simple Join (Medium)

**Task:** List the names of pets and the names of their owners.

**Setup Code (Run in CLI):**

```python
# Assumes 'pets' table exists from Puzzle 1
/create owners
owners['owner_id'] = [101, 102, 103, 104, 105]
owners['name'] = ['Alice', 'Bob', 'Carol', 'David', 'Eve']
owners['pet_id'] = [1, 2, 3, 4, 6] # Note: Pet ID 5 has no owner, Pet ID 6 doesn't exist in pets
```

**Expected Output:** A table showing pet names alongside their owner's name for pets 1 through 4. Buddy/Alice, Lucy/Bob, Max/Carol, Daisy/David.

---

## Puzzle 5: Finding Max Value in Groups (Medium/Hard)

**Task:** Find the name and salary of the employee with the highest salary in each department.

**Setup Code (Run in CLI):**

```python
/create employees
employees['emp_id'] = [1, 2, 3, 4, 5, 6, 7]
employees['name'] = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace']
employees['department'] = ['HR', 'Engineering', 'Engineering', 'HR', 'Sales', 'Sales', 'Engineering']
employees['salary'] = [60000, 80000, 85000, 65000, 70000, 75000, 90000]
```

**Expected Output:** A table showing the highest earner(s) from each department:
*   HR: David, 65000
*   Engineering: Grace, 90000
*   Sales: Frank, 75000

---

## Puzzle 6: Self Join or Subquery (Hard)

**Task:** Find the names of employees who earn more than their managers. Assume direct managers are listed in a separate table.

**Setup Code (Run in CLI):**

```python
# Assumes 'employees' table exists from Puzzle 5
/create managers
managers['emp_id'] = [1, 2, 3, 5, 6, 7] # Alice, Bob, Charlie, Eve, Frank, Grace
managers['manager_id'] = [4, 7, 7, -1, 5, 2] # David, Grace, Grace, NULL (Top), Eve, Bob
# Map: Alice->David, Bob->Grace, Charlie->Grace, Eve->(None), Frank->Eve, Grace->Bob
```


**Expected Output:** A table showing employees who earn more than their manager. Based on the data:
*   Grace (90000) earns more than Bob (80000).
*   Bob (80000) does *not* earn more than Grace (90000).
*   Frank (75000) earns more than Eve (70000).

---

## Puzzle 7: Outer Join (Medium)

**Task:** List all employees and their corresponding manager's name. Include employees who do not have a manager listed in the `managers` table.

**Setup Code:** (Use `employees` and `managers` tables from Puzzle 5 & 6)


**Expected Output:** A table showing each employee's name and their manager's name. Employees without managers (like Eve) should still appear, perhaps with a NULL value for the manager's name.

---

## Puzzle 8: Conditional Aggregation (Medium)

**Task:** Count the number of employees in the 'Engineering' department versus the total number of employees in all *other* departments combined.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A table showing two counts, one for 'Engineering' employees and one for 'Other Departments'. (e.g., Engineering: 3, Other: 4)

---

## Puzzle 9: Working with Dates (Medium)

**Task:** Find employees hired in the year 2023.

**Setup Code (Run in CLI):**

```python
# Assumes 'employees' table exists from Puzzle 5
employees['hire_date'] = ['2022-05-10', '2023-01-15', '2023-08-20', '2022-11-01', '2024-02-28', '2023-06-10', '2022-03-15']
```

**Expected Output:** A table listing the names of employees hired in 2023 (Bob, Charlie, Frank).

---

## Puzzle 10: String Matching (Easy)

**Task:** Find all employees whose names start with the letter 'A'.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A table listing the employee(s) whose name starts with 'A' (Alice).

---

## Puzzle 11: Multiple Joins (Medium)

**Task:** List the names of projects assigned to employees working in the 'Engineering' department.

**Setup Code (Run in CLI):**

```python
# Assumes 'employees' table exists from Puzzle 5
/create projects
projects['proj_id'] = ['P1', 'P2', 'P3', 'P4']
projects['proj_name'] = ['Website Redesign', 'Database Migration', 'Mobile App', 'Security Audit']

/create assignments
assignments['assign_id'] = [101, 102, 103, 104, 105, 106]
assignments['emp_id'] = [2, 3, 7, 1, 3, 5] # Bob, Charlie, Grace, Alice, Charlie, Eve
assignments['proj_id'] = ['P1', 'P1', 'P2', 'P3', 'P4', 'P3']
```

**Expected Output:** A table listing the project names associated with 'Engineering' employees (Bob, Charlie, Grace). Expected projects: Website Redesign, Database Migration, Security Audit.

---

## Puzzle 12: Finding Duplicates (Medium)

**Task:** Find all salary values that are paid to more than one employee.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A table listing salary values that appear more than once. (In the sample data, no salary is duplicated, but the query should work if there were duplicates). If Alice also made 80000, the output should show 80000.

---

## Puzzle 13: Nth Highest Value (Hard)

**Task:** Find the name and salary of the employee with the 3rd highest salary. (Assume no ties for simplicity, or define how to handle ties).

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A table showing the name and salary of the 3rd highest paid employee (Bob, 80000).

---

## Puzzle 14: Pivot Data (Hard)

**Task:** Display the total salary budget for each department, but with departments as columns rather than rows.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A single row table with columns like `HR_Total`, `Engineering_Total`, `Sales_Total`, showing the sum of salaries for each. (HR: 125000, Engineering: 255000, Sales: 145000)

---

## Puzzle 15: Set Operations (Easy)

**Task:** List the names of all employees who are in the 'HR' department *or* the 'Sales' department.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A table listing the names of employees in HR or Sales (Alice, David, Eve, Frank).

---

## Puzzle 16: Correlated Subquery (Hard)

**Task:** For each employee, find the average salary of *their specific department*.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** A table showing each employee's name, department, salary, and a column with the average salary for their department (e.g., Alice, HR, 60000, 62500; Bob, Engineering, 80000, 85000; etc.).

---

## Puzzle 17: Feature Engineering - Binning (Medium)

**Task:** Categorize customers into 'Low', 'Medium', or 'High' spenders based on their total purchase amount. Thresholds: Low < 100, Medium < 500, High >= 500.

**Setup Code (Run in CLI):**

```python
/create customers
customers['customer_id'] = [1, 2, 3, 4, 5, 6]
customers['name'] = ['Liam', 'Olivia', 'Noah', 'Emma', 'Oliver', 'Ava']

/create purchases
purchases['purchase_id'] = [101, 102, 103, 104, 105, 106, 107, 108]
purchases['customer_id'] = [1, 2, 1, 3, 4, 2, 5, 6]
purchases['amount'] = [50.0, 120.0, 30.0, 450.0, 600.0, 80.0, 25.0, 700.0]
```

**Expected Output:** A table showing each customer's ID, name, total purchase amount, and their spending category ('Low', 'Medium', or 'High').

---

## Puzzle 18: Feature Engineering - Lag Features (Hard)

**Task:** For each product on each date, find the sales amount from the *previous* day. Assume dates are consecutive where data exists.

**Setup Code (Run in CLI):**

```python
/create daily_sales
daily_sales['sale_date'] = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03']
daily_sales['product_id'] = [10, 10, 10, 20, 20, 20]
daily_sales['sales_amount'] = [200, 210, 215, 70, 75, 80]
# Convert date string to actual date type if needed by the specific SQL environment
# In sqlite-web or similar, date strings might work directly for comparisons
# If not, you might need: daily_sales['sale_date'] = pd.to_datetime(daily_sales['sale_date']) # before loading - anyway this works fine for the current sqlite3 environment
```

**Expected Output:** A table showing sale date, product ID, sales amount, and the previous day's sales amount (or NULL/0 for the first day).

---

## Puzzle 19: Data Cleaning - Identifying Missing Values (Easy)

**Task:** Find all rows in the `sensor_readings` table where the `temperature` reading is missing (NULL).

**Setup Code (Run in CLI):**

```python
/create sensor_readings
sensor_readings['reading_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sensor_readings['sensor_id'] = ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'B', 'A', 'C', 'B', 'A']
sensor_readings['timestamp'] = ['2023-05-01 10:00', '2023-05-01 10:00', '2023-05-01 10:05', '2023-05-01 10:00', '2023-05-01 10:05', '2023-05-01 10:10', '2023-05-02 10:00', '2023-05-02 10:00', '2023-05-02 10:05', '2023-05-02 10:00', '2023-05-02 10:05', '2023-05-04 10:10']
sensor_readings['temperature'] = [25.5, 22.0, None, 30.1, None, 26.0, 28.0, 29.2, 27.3, None, 28.5, 27.8] # Using Python's None for NULL
sensor_readings['humidity'] = [60, 65, 62, 55, 68, 61, 63, 64, 66, 67, 69, 90]

**Expected Output:** A table showing the full rows (all columns) for readings where the temperature is NULL (reading_ids 3 and 5).

---

## Puzzle 20: Data Sampling (Medium)

**Task:** Select approximately 50% of the user sessions randomly from the `user_sessions` table.

**Setup Code (Run in CLI):**

```python
/create user_sessions
user_sessions['session_id'] = range(1, 21) # 20 sessions
user_sessions['user_id'] = [ (i % 5) + 1 for i in range(20)] # Users 1-5
user_sessions['duration_seconds'] = [ (i * 10) + 30 for i in range(20)]
```

**Expected Output:** A table containing roughly half (around 10 rows) of the original `user_sessions` table, selected randomly. The exact rows will vary on each execution.

---

## Puzzle 21: A/B Testing Analysis - Basic Comparison (Medium)

**Task:** Calculate the conversion rate (percentage of users who made a purchase) for users in the 'Control' group versus the 'Treatment' group.

**Setup Code (Run in CLI):**

```python
/create experiment_users
experiment_users['user_id'] = range(1, 11) # 10 users
experiment_users['group_name'] = ['Control'] * 5 + ['Treatment'] * 5

/create user_actions
user_actions['action_id'] = range(101, 116) # 15 actions
user_actions['user_id'] = [1, 2, 6, 3, 7, 8, 1, 4, 9, 6, 2, 10, 5, 7, 3] # Some users repeat actions
user_actions['action_type'] = ['view', 'view', 'view', 'click', 'view', 'purchase', 'click', 'view', 'purchase', 'click', 'purchase', 'view', 'view', 'purchase', 'purchase']
```

**Expected Output:** A table showing two rows, one for 'Control' and one for 'Treatment', with their respective conversion rates. (Control: User 3 & 5 purchased? 2 / 5 users. Treatment: User 7, 8, 9 purchased? 3 / 5 users).

---

## Puzzle 22: Time Series Analysis - Simple Moving Average (Hard)

**Task:** Calculate the 3-day moving average of website `visits` for the entire site.

**Setup Code (Run in CLI):**

```python
/create website_traffic
website_traffic['log_date'] = ['2023-03-01', '2023-03-02', '2023-03-03', '2023-03-04', '2023-03-05', '2023-03-06', '2023-03-07']
website_traffic['visits'] = [1000, 1100, 1050, 1200, 1150, 1300, 1250]
# Ensure log_date is treated as a date for ordering
```

**Expected Output:** A table showing each date and the corresponding 3-day moving average of visits. The first two days will have averages based on fewer than 3 days.

---

## Puzzle 23: User Segmentation - RFM Prep (Medium/Hard)

**Task:** For each customer, calculate their total number of purchases (Frequency) and the total amount spent (Monetary Value). (Recency is harder without transaction dates, so we skip it here).

**Setup Code:** (Use `customers` and `purchases` tables from Puzzle 11)

**Expected Output:** A table showing `customer_id`, `name`, `Frequency` (count of purchases), and `Monetary` (sum of purchase amounts) for each customer.

---

## Puzzle 24: Anomaly Detection - Simple Thresholding (Medium)

**Task:** Identify users whose average session duration is more than 1.5 times the overall average session duration across all users.

**Setup Code:** (Use `user_sessions` table from Puzzle 14)
Adjust the avg duration for user 1 to be only 10 and user 5 to be 250:
```python
user_sessions.loc[user_sessions.user_id == 1,'duration_seconds'] = 10
user_sessions.loc[user_sessions.user_id == 5,'duration_seconds'] = 250
```

**Expected Output:** A table listing the `user_id` and their average session duration for users identified as having anomalously long average sessions.

---

## Puzzle 25: Data Transformation - Finding Min/Max for Scaling (Easy)

**Task:** Find the minimum and maximum values for the `feature1` column in the `ml_data` table. This is often needed for Min-Max scaling: `(value - min) / (max - min)`.

**Setup Code (Run in CLI):**

```python
/create ml_data
ml_data['id'] = range(1, 8)
ml_data['feature1'] = [10.5, 12.1, 8.3, 15.0, 9.9, 11.2, 14.5]
ml_data['feature2'] = [100, 110, 95, 120, 105, 108, 115]
ml_data['label'] = [0, 1, 0, 1, 0, 1, 1]
```

**Expected Output:** A single row table showing the minimum and maximum values found in `feature1`. (Min: 8.3, Max: 15.0).

**Follow-up:** If you want to scale the `feature1` column, you can use the formula: `(value - min) / (max - min)`.

---

## Puzzle 26: Model Evaluation Prep - Confusion Matrix Counts (Hard)

**Task:** Given a table of actual labels and predicted labels, count the number of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). Assume '1' is the positive class and '0' is the negative class.

**Setup Code (Run in CLI):**

```python
/create predictions
predictions['id'] = range(1, 11) # 10 predictions
predictions['actual_label'] = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
predictions['predicted_label'] = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0]
```

**Expected Output:** A single row table with four columns: TP, FP, TN, FN, showing the counts for each category based on the provided data.

---

## Puzzle 27: Handling Missing Values - Imputation (Medium)

**Task:** Replace missing `temperature` values in the `sensors2` table with the average temperature for the same sensor on the same day.

**Setup Code:** 
```python
/create sensors2
sensors2['reading_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sensors2['sensor_id'] = ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'B', 'A', 'C', 'B', 'A']
sensors2['timestamp'] = ['2023-05-01 10:00', '2023-05-01 10:00', '2023-05-01 10:05', '2023-05-01 10:00', '2023-05-01 10:05', '2023-05-01 10:10', '2023-05-02 10:00', '2023-05-02 10:00', '2023-05-02 10:05', '2023-05-02 10:00', '2023-05-02 10:05', '2023-05-04 10:10']
sensors2['temperature'] = [25.5, 22.0, None, 30.1, None, 26.0, 28.0, 29.2, 27.3, None, 28.5, 27.8] # Using Python's None for NULL
sensors2['humidity'] = [60, 65, 62, 55, 68, 61, 63, 64, 66, 67, 69, 90]

**Expected Output:** The modified `sensors2` table with missing `temperature` values replaced.

---

## Puzzle 28: Data Quality Check - Duplicate Rows (Easy)

**Task:** Identify and count duplicate rows in the `user_sessions` table.

**Setup Code:** (Use `user_sessions` table from Puzzle 14) and add a duplicate row:
```python
user_sessions.loc[(len(user_sessions) + 1)] = [1, 1, 10]
```

**Expected Output:** A table showing the duplicate rows and their counts.

---

## Puzzle 29: Data Normalization - Scaling (Medium)

**Task:** Scale the `feature1` and `feature2` columns in the `ml_data` table using Min-Max scaling.

**Setup Code:** (Use `ml_data` table from Puzzle 25)

**Expected Output:** The `ml_data` table with `feature1` and `feature2` scaled between 0 and 1.

---

## Puzzle 30: Data Visualization Prep - Aggregating for Plotting (Medium)

**Task:** Prepare data for plotting the distribution of `feature1` values in the `ml_data` table by aggregating into bins.

**Setup Code:** (Use `ml_data` table from Puzzle 25)

**Expected Output:** A table showing the bins and their respective counts, ready for plotting.

---

## Puzzle 31: Sales Rep Ranking (Medium)

**Task:** Using the `sales` table, rank each sales representative by their total sales amount, showing the `rep_id`, `total_sales`, and assign a `sales_rank`.

**Setup Code (Run in CLI):**

```python
/create sales
sales['sale_id'] = [1, 2, 3, 4, 5, 6]
sales['rep_id'] = [101, 102, 101, 103, 102, 101]
sales['amount'] = [500, 300, 200, 400, 600, 150]
```

**Expected Output:** A table with columns `rep_id`, `total_sales`, and `sales_rank`.

---
## Puzzle 32: Cumulative Monthly Revenue (Medium)

**Task:** Calculate cumulative revenue month over month using the `sales_series` table. Include `month`, `revenue`, and `cumulative_revenue`.

**Setup Code:** 
```python
/create sales_series
sales_series['month'] = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12']
sales_series['amount'] = [1200, 1500, 1300, 0, 1600, 1400, 1700, 1800, 1550, 1650, 1750, 1850]
```

**Expected Output:** A table sorted by `month` showing `cumulative_revenue`.

---
## Puzzle 33: Top N Employees by Department (Medium)

**Task:** Find the top 2 highest paid employees in each department using the `employees2` table. Return `department`, `employee_name`, and `salary`.

**Setup Code:**
```python
/create employees2
employees2['emp_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
employees2['name'] = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack']
employees2['department'] = ['HR', 'Engineering', 'Engineering', 'HR', 'Sales', 'Sales', 'Engineering', 'HR', 'Engineering', 'Sales']
employees2['salary'] = [60000, 80000, 85000, 65000, 70000, 75000, 90000, 30000, 45000, 25000]
```

**Expected Output:** The top 2 earners per department.

---
## Puzzle 34: Employee-Manager Self-Join (Easy)

**Task:** List each employee along with their manager's name using the `employees` table from Puzzle 5.

**Setup Code:** (Use `employees` table from Puzzle 5)

**Expected Output:** Columns `employee_id`, `employee_name`, and `manager_name`.

---
## Puzzle 35: Organizational Hierarchy (Hard)

**Task:** Show the full management chain for each employee using the `employees2` table from Puzzle 33 and the managers2 table defined below.

**Setup Code:** (Use `employees2` table from Puzzle 33 and the managers2 table defined below)
```python
/create managers2
managers2['emp_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
managers2['manager_id'] = [2, 7, 6, 2, 6, 7, None, 1, 3, 5]
```

**Expected Output:** A table listing `employee_id`, `employee_name`, and the full path of managers above them.

---
## Puzzle 36: Customers Without Purchases (Easy)

**Task:** Identify customers who have never made a purchase from the `customers` and `purchases` tables used in Puzzle 9.

**Setup Code:** (Use `customers` and `purchases` tables from Puzzle 9)
Add customers who haven't made purchases:
customers.loc[(len(customers) + 1)] = [7, 'Bob']
customers.loc[(len(customers) + 1)] = [8, 'Mary']

**Expected Output:** A list of customer IDs or names with no purchases.

---
## Puzzle 37: Common Products (Easy)

**Task:** Find products that appear in both the `product_sales` table and a `returns` table.

**Setup Code:** 

```python
/create product_sales
product_sales['product_id'] = [101, 102, 103, 104, 102]
product_sales['quantity'] = [10, 20, 30, 40, 15]

/create returns
returns['return_id'] = [1, 2, 3]
returns['product_id'] = [102, 104, 106]
returns['quantity'] = [2, 5, 1]
```

**Expected Output:** A list of `product_id`s present in both tables.

---
## Puzzle 38: Monthly Sales Pivot (Medium)

**Task:** Pivot the `product_sales2` table so months become columns and rows show `product_id` with total sales per month.

**Setup Code:** 

```python
/create product_sales2
product_sales2['month'] = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
product_sales2['product_id'] = [101, 102, 103, 104, 102, 103]
product_sales2['quantity'] = [10, 20, 30, 40, 15, 20]
product_sales2['price'] = [1, 2, 3, 4, 2, 3]
```

**Expected Output:** A pivoted table with `product_id` and monthly sales columns.

---
## Puzzle 39: Survey Data Unpivot (Medium)

**Task:** Unpivot a `survey` table with columns `response_id`, `q1`, `q2`, and `q3` into a tall format.

**Setup Code:** (Create `survey` table with sample answers for `q1`, `q2`, and `q3`.)

```python
/create survey
survey['response_id'] = [101, 102, 103, 104]
survey['q1'] = ['High', 'Medium', 'Low', 'High']      # satisfaction level
survey['q2'] = ['Yes', 'No', 'Yes', 'No']            # would recommend
survey['q3'] = ['Daily', 'Weekly', 'Monthly', 'Daily']  # usage frequency
```

**Expected Output:** Columns `response_id`, `question`, and `answer`.

---
## Puzzle 40: JSON Data Extraction (Medium)

**Task:** Extract nested values from a JSON column in a `json_data` table, e.g., get `user.id` and `order.total`.

**Setup Code:** (Create `json_data` table with a `data` JSON column.)

```python
/create json_data
json_data['id'] = [1, 2, 3, 4]
json_data['data'] = [
    '{"user": {"id": 101, "name": "John"}, "order": {"total": 100}}',
    '{"user": {"id": 102, "name": "Jane"}, "order": {"total": 200}}',
    '{"user": {"id": 103, "name": "Bob"}, "order": {"total": 300}}',
    '{"user": {"id": 104, "name": "Alice"}, "order": {"total": 400}}'
]
```

**Expected Output:** A table with `user_id` and `order_total`.

---
## Puzzle 41: Full-Text Search (Medium)

**Task:** Find articles containing the keyword "database" in an `articles` table.

**Setup Code:** 

```python
/create articles
articles['article_id'] = [1, 2, 3, 4, 5]
articles['content'] = ['database management', 'design', 'database optimization', 'warehouse performance', 'internal database security']
```

**Expected Output:** Rows of `article_id` and a snippet of `content`.

---
## Puzzle 42: Moving Average Excluding Current Row (Hard)

**Task:** Calculate a 3-point moving average in a `measurements` table, excluding the current row.

**Setup Code:** 

```python
/create measurements
measurements['reading_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
measurements['value'] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
```

**Expected Output:** Columns `reading_id`, `value`, and `moving_avg`.

---
## Puzzle 43: Top Percentile Employees (Medium)

**Task:** Identify employees in the top 10% salary bracket from the `employees2` table in Puzzle 33.

**Setup Code:** (Use `employees2` table from Puzzle 33.)

**Expected Output:** A list of `employee_id` and `salary` for the top 10%.

---
## Puzzle 44: Conditional Aggregation by Status (Easy)

**Task:** Count the number of orders in each `status` category from the `order_status` table.

**Setup Code:**

```python
/create order_status
order_status['order_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
order_status['status'] = ['Shipped', 'Pending', 'Shipped', 'Delivered', 'Pending', 'Shipped', 'Delivered', 'Shipped', 'Pending', 'Delivered']
```

**Expected Output:** Counts for each status value.

---
## Puzzle 45: Latest Order Per Customer (Hard)

**Task:** For each customer, show their most recent order using `customers` and `purchases` tables from Puzzle 17.

**Setup Code:** (Use `customers` and `purchases` tables.)
Add an order_date column to purchases table
```python
purchases['order_date'] = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08']
```

**Expected Output:** Columns `customer_id`, `customer_name`, and latest `order_id`.

---
## Puzzle 46: Geospatial Distance Filter (Hard) - maybe too difficult

**Task:** Given a `locations` table with `lat` and `lon`, find points within 10 km of a reference point.

**Setup Code:** (Create `locations` table with sample latitude/longitude.)

```python
/create locations
locations['id'] = [1, 2, 3, 4, 5]
locations['lat'] = [40.7128, 34.0522, 37.7749, 47.6062, 33.7489]
locations['lon'] = [-74.006, -118.2437, -122.4194, -122.3321, -118.2437]
```

**Expected Output:** Rows of locations within the radius.

---
## Puzzle 47: Inactive Customers (Easy)

**Task:** Find customers who have not logged in for over 30 days using a `last_login` column in `customers`.

**Setup Code:** (Use `customers` table with `last_login` date.)

```python
# Add a last_login column to customers table
customers['last_login'] = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2025-04-29']
```

**Expected Output:** List of `customer_id` and `last_login` older than 30 days.

---
## Puzzle 48: Data Masking (Medium)

**Task:** Mask the middle digits of a Social Security number in a `users_with_PII` table, showing only the first and last two digits.

**Setup Code:** (Create `users_with_PII` table with `user_id` and `ssn`.)

```python
/create users_with_PII
users_with_PII['user_id'] = range(1, 11)
# Generate some plausible SSN-like strings (not real SSNs)
import random
ssns = [f'{random.randint(100, 999):03}-{random.randint(10, 99):02}-{random.randint(1000, 9999):04}' for _ in range(10)]
users_with_PII['ssn'] = ssns
```

**Expected Output:** Columns `user_id` and `masked_ssn`.

---
## Puzzle 49: Dense Ranking Scores (Medium)

**Task:** Assign a dense rank to players in a `scores` table based on their `score`, showing ties without gaps.

**Setup Code:** (Create `scores` table with `player_id` and `score`.)

```python
/create scores
scores['player_id'] = range(1, 11)
scores['score'] = [850, 920, 780, 920, 850, 600, 780, 990, 700, 920] # Include ties
```

**Expected Output:** Columns `player_id`, `score`, and `dense_rank`.

---
## Puzzle 50: Net Sales Calculation with CTEs (Hard)

**Task:** Calculate net sales by subtracting returns from total sales using CTEs on `product_sales` and `returns` tables.

**Setup Code:** (Use `product_sales` table from Puzzle 38 and `returns` table from Puzzle 37.)

**Expected Output:** Columns `product_id` and `net_sales`.

---
## Puzzle 51: Get counts of unique values for each column (Easy)

**Task:** For a given table, return the count of unique values for each column.

**Setup Code:** (Create `sales` table from Puzzle 12.)

**Expected Output:** A table with columns `column_name` and `unique_count`.

---

## Puzzle 52: Given a table with columns a, b, and c, find the number of non-null values in each column

**Task:** For a given table, return the count of non-null values for each column.

**Setup Code:** (Load `sensor_readings` table from above.)

**Expected Output:** A table with columns `column_name` and `non_null_count`.

---

## Puzzle 53: Differences between tables (Easy)

**Task:** Given two tables C and D, count the number of rows whose id occurs only in C and not D, and vice versa (D and not C).

**Setup Code:** (Create tables `C` and `D`.)

```python
# Run these lines one by one in the CLI
/create C
C['id_col'] = [1, 2, 3, 4, 5, 9, 9, 8] # 5 rows w/ id only in C
C['value'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

/create D
D['id_col'] = [4, 5, 6, 7, 8, 11, 11, 12, 13, 14, 15] # 8 rows w/ id only in D
D['value'] = ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
```

**Expected Output:** A table with columns `id_col` and `value` showing the unique rows in each table.

Schema C: id_col (int), value (string)

Schema D: id_col (int), value (string)


## Puzzle 54: Imputation

**Task:** Given a table with columns `id`, `education`, and `income`, impute the missing values of `income` using the average income of people with the same education level.

**Setup Code:** (Create `imputation_table` with columns `id`, `education`, and `income`.)

```python
# Run these lines one by one in the CLI
/create imputation_table
imputation_table['id'] = [1, 2, 3, 4, 5]
imputation_table['education'] = [12, 16, 12, 13, 17]
imputation_table['income'] = [0.45, 0.65, None, 0.85, None]
```

**Expected Output:** A table with columns `id`, `education`, and `income` with imputed values.
