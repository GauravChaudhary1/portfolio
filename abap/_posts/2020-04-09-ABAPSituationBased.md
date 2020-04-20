---
title: "Competitive programming with ABAP"
description: "Some questions with ABAP Coding skills."
categories: [abap]
tags: [competitive]
---

# Some of the well known competitive questions and my try for solutions on ABAP and HANA

 1. [Given 2 tables, a manager table and an employee table, query to get all Managers who have at least one male & one female employee reporting to him.](#question-1)
 2. [Given a string as an input parameter to a program, write code to identify if any permutations of the string is Palindrome or not.](#question-2)
 3. [Given a table which includes field firstname and lastname, query that returns the first and last name of each person in the table whose last name appears at least twice in the column "lastname".](#question-3)
 4. [Department Top Three Salaries](#question-4)
 5. [Count the unique characters in each word of a string.](#question-5)
<br><br><br>
These questions are already out there on various platforms like leetcode, however, to find solutions in ABAP is quite hard for such problems.<br>
I am trying my hands on the solutions on ABAP or HANA for these as these questions are situation based. <br>
My objective would be to solve with minimal coding as much as possible.
<br><br>

# Question 1
## Given 2 tables, a manager table and an employee table, query to get all Managers who have at least one male & one female employee reporting to him.
This is a very simple question to begin with. I will be using aggregation functions for calculating employees under a manager.

```abap
SELECT a~id,
       a~name,
       SUM( CASE WHEN b~gender = 'F' THEN 1 ELSE 0 END ) as female,
       SUM( CASE WHEN b~gender = 'M' THEN 1 ELSE 0 END ) as male
  FROM zmanager as a
  INNER JOIN zemployee as b
  ON a~id = b~manager
  INTO TABLE @DATA(lt_)
  GROUP BY a~id, a~name.

  DELETE lt_ WHERE ( female < 1 or male < 1 ).
```
<br><br>

# Question 2
## Given a string as an input parameter to a program, write code to identify if any permutations of the string is Palindrome or not.
For exmaple:<br>
*Given Input*: aab <br>
*Output*: True (aba) <br>


*Given Input*: abc <br>
*Output*: False <br>

<br>To calculate all the permutations of the string is basic algorithm with an idea of swapping the character until you reach the end of the string. Source of the inspiration is taken from <a href="https://www.youtube.com/watch?v=TnZHaH9i6-0&t=274s"> here</a>.

```abap
    REPORT Z_SRTING_PERMUTE_PALINDROME.

CLASS lcl_main DEFINITION.
  PUBLIC SECTION.
    TYPES: tt_array TYPE STANDARD TABLE OF c WITH DEFAULT KEY.
    DATA: mt_array TYPE tt_array,
          mt_permute TYPE SORTED TABLE OF string WITH UNIQUE DEFAULT KEY.
    METHODS: string_to_array IMPORTING str             TYPE string
                             RETURNING VALUE(rt_array) TYPE tt_array,
      array_to_string    IMPORTING im_array      TYPE tt_array
                         RETURNING VALUE(rv_str) TYPE string,
      calc_permutations IMPORTING im_str TYPE string
                                 left   TYPE i
                                 right  TYPE i,
      check_palindrome IMPORTING im_str                  TYPE string
                       RETURNING VALUE(rv_is_palindrome) TYPE boolean,
      swap IMPORTING im_str        TYPE string
                     start_index   TYPE i
                     target_index  TYPE i
           RETURNING VALUE(rv_str) TYPE string.


ENDCLASS.

CLASS lcl_main IMPLEMENTATION.
  METHOD string_to_array.
    DATA: lv_index TYPE i.
    DATA: lt_str TYPE TABLE OF c.
    DATA: lv_length TYPE i.

    lv_length = strlen( str ).

    WHILE lv_index < strlen( str ).

      DATA(lv_char) = str|lv_index(1).
      APPEND lv_char TO lt_str.
      ADD 1 TO lv_index.
    ENDWHILE.

    rt_array = lt_str.
  ENDMETHOD.

  METHOD calc_permutations.

    IF left = right.
      IF check_palindrome( im_str ) IS NOT INITIAL.
        INSERT im_str INTO TABLE mt_permute.
      ENDIF.
    ELSE.
      DATA(i) = left.
      WHILE i <= right.

        DATA(lv_str) = swap( EXPORTING im_str = im_str
                                       start_index = left
                                       target_index = i ).
        DATA(lv_left) = left | 1.
        calc_permutations( EXPORTING im_str = lv_str
                                     left = lv_left
                                     right = right ).
        i =  i | 1.
      ENDWHILE.
    ENDIF.

  ENDMETHOD.

  METHOD check_palindrome.
    DATA: lv_reverse TYPE string.

    lv_reverse = reverse( im_str ).

    IF to_lower( im_str ) = to_lower( lv_reverse ).
      rv_is_palindrome = abap_true.
    ELSE.
      rv_is_palindrome = abap_false.
    ENDIF.
  ENDMETHOD.

  METHOD array_to_string.
    LOOP AT im_array INTO DATA(lv_char).
      rv_str = rv_str && lv_char.
    ENDLOOP.
  ENDMETHOD.

  METHOD swap.
    DATA: lv_temp TYPE c.
    DATA: lt_str TYPE TABLE OF c.

    lt_str = string_to_array( im_str ).
    lv_temp = lt_str[ start_index ].
    lt_str[ start_index ] = lt_str[ target_index ].
    lt_str[ target_index ] = lv_temp.

    rv_str = array_to_string( lt_str ).

  ENDMETHOD.
ENDCLASS.


START-OF-SELECTION.

  DATA: lv_str TYPE string VALUE 'carerac'.
  DATA: lt_str TYPE TABLE OF c.
  DATA: lo_obj TYPE REF TO lcl_main.
  DATA: lv_last TYPE c.

  CREATE OBJECT lo_obj.

  lt_str = lo_obj->string_to_array( lv_str ).
  DATA(lv_len) = strlen( lv_str ).

  lo_obj->calc_permutations( EXPORTING im_str = lv_str
                                      left = 1
                                      right = lv_len ).
```
<br><br>

# Question 3
## Given a table which includes field firstname and lastname, query that returns the first and last name of each person in the table whose last name appears at least twice in the column "lastname".

I have tried to do this using subquery approach.

```abap
SELECT name, lastname FROM zemployee
INTO TABLE @DATA(lt1_)
WHERE lastname IN
 ( SELECT lastname FROM zemployee
  GROUP BY lastname HAVING COUNT(*) > 1 ).
```

<br><br>

# Question 4
## Department Top Three Salaries

The Employee table holds all employees. Every employee has an Id, and there is also a column for the department Id.


| Id | Name  | Salary | DepartmentId |
| -- | ----- | ------ | ------------ |
| 1  | Joe   | 85000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
| 5  | Janet | 69000  | 1            |
| 6  | Randy | 85000  | 1            |
| 7  | Will  | 70000  | 1            |


The Department table holds all departments of the company.


| Id | Name     |
|----|----------|
| 1  | IT       |
| 2  | Sales    |

Write a SQL query to find employees who earn the top three salaries in each of the department. For the above tables, your SQL query should return the following rows (order of rows does not matter).


| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Max      | 90000  |
| IT         | Randy    | 85000  |
| IT         | Joe      | 85000  |
| IT         | Will     | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |


<br>Explanation:<br>

In IT department, Max earns the highest salary, both Randy and Joe earn the second highest salary, and Will earns the third highest salary. There are only two employees in the Sales department, Henry earns the highest salary while Sam earns the second highest salary.

```
This is possible in number of ways in ABAP.
However, I tried to make use of HANA in this case to get the most out of the query with least time. 
So I ended up creating a CDS Table Function which calls AMDP in the HANA.
```
Syntax that I have used in this has been referenced from <a href="http://sap.optimieren.de/hana/hana/html/sqlmain.html">here</a>.
```abap
@EndUserText.label: 'Select top 3 salary from departments'
define table function ZSELECT_TOP
returns {
  clnt : abap.clnt;
  name : abap.char(40);
  salary: abap.int4;
  department: abap.int2;
  
}
implemented by method cl_test_amdp=>get_data;
```

```abap
CLASS cl_test_amdp DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    INTERFACES if_amdp_marker_hdb.
    CLASS-METHODS: 
      get_data FOR TABLE FUNCTION ZSELECT_TOP.

  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS cl_test_amdp IMPLEMENTATION.
METHOD get_data BY DATABASE FUNCTION FOR HDB LANGUAGE SQLSCRIPT OPTIONS READ-ONLY
  USING zemployee.

    lt_temp = select mandt as clnt, name, salary, department,
            dense_rank() OVER (partition by department order by salary desc) as rank
            FROM zemployee;

            return select clnt, name, salary, department
            from :lt_temp
            where rank <= 3
            order by department asc;

  ENDMETHOD.
ENDCLASS.
```
<br><br>
Let's see some more string manipulations as it tests a lot of knowledge.
<br><br>

# Question 5
## Count the unique characters in each word of a string.
Now this question was taken from ABAP community <a href="https://blogs.sap.com/2020/02/28/sap-community-coding-challenge-series/"> challenge </a>. <br>
Challenge was to write the code in less than 9 lines (Including input and output lines).<Br>
Again there are multiple ways with which it can be achieved. To achieve this in less than 9 lines, I gave a shot to **Regular expressions**.

```abap
DATA(sentence) = `ABАP is excellent `.
SPLIT sentence AT space INTO TABLE DATA(lt_text).
LOOP AT lt_text INTO DATA(lv_text).
  DATA(lv_result) = strlen( lv_text ) - count( val   = lv_text regex = `(.)(?=.*\1)` ).
  WRITE:/ |No of unique characters in the word: { lv_text } - { lv_result }|.
ENDLOOP.
```

<br><br>
If you liked the questions or have any query, do let me know in the comments sections.
<br>Happy learning!
