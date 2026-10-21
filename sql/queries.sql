
-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500, group by customer,
-- keep customers whose total exceeds 8000.
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Three-table JOIN
-- Revenue by city and product category (completed orders only).
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Scalar subquery
-- Products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- Q4: LEFT JOIN
-- Customers who have no completed order.
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Common Table Expression
-- Customers whose total completed spend is above average.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total
    FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend
FROM customer_spend cs, avg_spend a
WHERE cs.total_spend > a.avg_total
ORDER BY cs.total_spend DESC;


-- Q6: Ranking window functions
-- For each customer, compute total spend and RANK(), DENSE_RANK(), ROW_NUMBER()
-- partitioned by city, ordered by total_spend DESC.
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM (
    SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
) sub
ORDER BY city, total_spend DESC;


-- Q7: Running total
-- Daily completed revenue with cumulative revenue.
SELECT day, revenue,
       SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
) daily
ORDER BY day;
