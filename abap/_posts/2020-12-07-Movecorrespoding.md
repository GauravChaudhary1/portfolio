---
title: "MOVE-CORRESPONDING itab"
description: "Move-Corresponding internal tables? Should we use it?"
categories: [abap]
tags: [new features, 740]
---

# Move-Corresponding internal tables? Should we use it?
\
With all New features in ABAP, a very much needed MOVE-CORRESPONDING for internal tables was introduced. It does reduce some line of code, if using plain MOVE-CORRESPONDING structures. But should we use it, just because it ease the coding?  
\
Have a look at below comparison for total count of entries **450,000**.  

**Internal Tables - Move Corresponding**

```abap
GET RUN TIME FIELD DATA(lv_1).

" Internal Tables - Move Corresponding
MOVE-CORRESPONDING lt_hrp1001 TO lt_data.

GET RUN TIME FIELD DATA(lv_2).

DATA(lv_3) = lv_2 - lv_1.
WRITE: /'Move-Corresponding table: ' ,lv_3.

```

**Work Area - Move Corresponding using Loop**

```abap
GET RUN TIME FIELD DATA(lv_4).

"Work Area - Move Corresponding using Loop.
LOOP AT lt_hrp1001 INTO DATA(ls_hrp1001).
  MOVE-CORRESPONDING ls_hrp1001 TO ls_data.
  APPEND ls_data TO lt_data.
ENDLOOP.

GET RUN TIME FIELD DATA(lv_5).

DATA(lv_6) = lv_5 - lv_4.
WRITE: /'Loop with Move Corresponding Struct: ' , lv_6.

```

**Individual Field Assignment using Loop**

```abap
GET RUN TIME FIELD DATA(lv_7).

" Individual Field Assignment using Loop.
LOOP AT lt_hrp1001 INTO ls_hrp1001.
  ls_data-otype = ls_hrp1001-otype.
  ls_data-objid = ls_hrp1001-objid.
  ls_data-plvar = ls_hrp1001-plvar.
  ls_data-rsign = ls_hrp1001-rsign.
  ls_data-relat = ls_hrp1001-relat.
  ls_data-istat = ls_hrp1001-istat.
  ls_data-priox = ls_hrp1001-priox.
  ls_data-begda = ls_hrp1001-begda.
  ls_data-endda = ls_hrp1001-endda.
  ls_data-varyf = ls_hrp1001-varyf.
  ls_data-seqnr = ls_hrp1001-seqnr.
  APPEND ls_data TO lt_data.
ENDLOOP.

GET RUN TIME FIELD DATA(lv_8).

DATA(lv_9) = lv_8 - lv_7.
WRITE: /'Loop with Field assignments: ' , lv_9.

```

**OUTPUT**
![](/images/ABAP/20201207/1.png)


# Explaination
Now, let's see how is it that new feature is taking as double as processing time. According to the SAP, processing for this statement happens in these steps:  

- Similar name components are searched.
- Data from source table is extracted sequentially, similar to Loop.
- Content of each row is assigned to corresponding field.
- Lastly, table keys and table indexes are updated.

First three steps have no additional cost however updating table keys and indexes is costly operation. Which is why this statement takes longer processing time.  
Now should we completey ignore this statement? I believe, this statement comes in handy when we are dealing Non-standard internal tables where updating of table keys happens everytime internal table is appended.  

I am open to discussion if there is any point which you find misinterpreted.

*Happy Learning*

