--basic
SELECT _column FROM _table;
SELECT DISTINCT(_column) FROM _table;
SELECT COUNT(_column) from _table;
SELECT _column, count(distinct(_col)) FROM _table
    WHERE _column != _val AND _column2 > _val2;
    ORDER BY _col DESC,_col2 ASC    -- _col need not be in select
    LIMIT 10    -- limit at last

    WHERE _val NOT BETWEEN low AND high -- <low OR >high; between date uses 00:00 time
    WHERE _val NOT IN (_val1, _val2)
    WHERE _val NOT LIKE '_her%' and ILIKE '__a_q%'        -- regex * is sql %, regex . is sql _;ilike is for case insensitive
-- aggregate
    avg(round),count,max,min,sum,... -- only in select or having; fetches only one col
    GROUP BY -- categorical
select col, max(_col)
from _table
where _col = _val  -- should not refer aggregated col
group by _col2  -- after from or where; either have to use an aggregate function, or same col should be used in select and groupby
having max(_col) > _val -- for where on group by

-- joins
select _col as c from _table -- alias; cant be used in having or such, because alias is done after query executes, hence inaccessible for having, ...
select _col2,_tab.col from 
    _tab inner join _tab2       -- join defaults to inner join
        on _tab.col = _tab2.col
inner join
full outer join
    where _tab.id is null or _tab2.id is null -- one use case...XOR
left outer join
    where _tab2.id is null -- one use case...exclusive
right outer join

    select...
union
    select...
-- 
SHOW ALL
SELECT NOW()
    TIMEOFDAY()
    CURRENT_DATE
    EXTRACT(YEAR FROM <col>)
    AGE(<col>)
    TO_CHAR(<col>,'<format>')
-- 
select <col>, <col> + <col2> ...
    -- math oper - numerical
    -- str oper - string
-- 
select <col> from <tab>
where <col2> > (<query>) -- any <cond> is fine
    -- exists - if <subquery> returns any entry
-- constraints
    -- data types
        ...
    -- on col
        not null
        unique
        primary key
        foreign key
        check
        exclusion
    -- on table
        check
        references
        unique
        primary key
create table <tab>(
    <id> type <type> primary key ,
    <col> type <type> not null,
    ...
    <id> serial <constraint>,
    table_constraint <constraint>
) inherits <tab2>;
create table <tab>(
    <id> serial primary key ,
    <col> varchar(50) unique not null,
    <fid> <type> references <tab>(<col>) -- error if inserting un-present key
);

insert into <tab>(<col>,...)
    values
        (<val1>,...),
        (<val2>,...),
insert into <tab>(<col>,...)
    select ...

update <tab>
    set <col> = <val>, ...
    where <cond>\
    returning <col>,... -- returns effected rows' cols
update <tab>
    set <col> = <tab2>.<val>, ...
    from <tab2>
    where...

delete from <tab>
using <tab2> -- delete using join
where ... -- if no where, delete all rows
returning <col>,...

alter table <tab>
    add column <col>
    drop column <col>
        drop column <col> if exists <col> -- skip error
        drop column <col> cascade -- remove dependencies?
    alter <col> set default <col> -- not null,drop not null, add constraint, ...
    rename to <newtab>
    rename <col> to <col>

create ...(
    <col> <type> check (<cond>)
)
--
select
    case <expr>     -- can be aggregated like sum(case...)
        when <cond> then <expr>
        ...
        else <expr>
    end as cased_col
from <tab>

select coalesce (null,1,2) -- returns first non null, here 1, can use for default value for nulls coalesce(<col>,<default>)
select cast(<col> as <type>)
nullif(<val1>,<val2>) -- null if val1 == val2

create view <view> as <query>
create or replace view <view> as <query>
drop view <view>
alter view <view> rename <newview>
-- py
import psycopg2 as pg2
conn = pg2.connect(database='', user='', password='')
cur = conn.cursor()
cur.execute(<query>)
cur.fetchone();...
conn.close()
-- asky
select distinct(_col1), _col2 as c,<expr>,rank(_col3)
    case
        when <cond> then <val>
    end as cased_col
from _tab
where <cond> -- exists <query>;any <query>;all <query>
group by _col1
having <cond>
order by <expr> ASC/DESC
limit 10 ;
-- 
    select <query> 
union all -- keep dups
^
except / minus
^
intersect
    select <query> 
-- 
    select from tab 
join
    (select from) tab2
on <cond>
-- window
-- func() over (partition by <col> order by <col>); for sum(), count(),... too
-- order by <col> rows between n preceding and m following
rank()
dense_rank()
row_number()
ntile()
lead(<col>,n)
lag(<col>,n)

-- ?
where _val = 'val' and max(_col3) > 30
