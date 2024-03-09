import React, { useState, useEffect } from 'react';
import './ReportAndAnalysis.css';

const ReportAndAnalysis = () => {
    // State to hold sales data
    const [salesData, setSalesData] = useState([]);

    // Fetch sales data from backend
    useEffect(() => {
        fetchSalesData();
    }, []);

    const fetchSalesData = async () => {
        try {
            // Make API call to fetch sales data
            const response = await fetch('/api/sales');
            const data = await response.json();
            setSalesData(data);
        } catch (error) {
            console.error('Error fetching sales data:', error);
        }
    };

    // Function to calculate total revenue
    const calculateTotalRevenue = () => {
        // Iterate through sales data and sum up revenue
        return salesData.reduce((total, sale) => total + sale.amount, 0);
    };

    // Render sales data
    const renderSalesData = () => {
        return (
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Amount</th>
                        {/* Add more columns as needed */}
                    </tr>
                </thead>
                <tbody>
                    {salesData.map((sale, index) => (
                        <tr key={index}>
                            <td>{sale.date}</td>
                            <td>Rs. {sale.amount.toFixed(2)}</td>
                            {/* Add more columns as needed */}
                        </tr>
                    ))}
                </tbody>
            </table>
        );
    };

    return (
        <div className="report-and-analysis">
            <h2>Sales Report and Analysis</h2>
            <div className="summary">
                <p>Total Revenue: ${calculateTotalRevenue().toFixed(2)}</p>
                {/* Add more summary information as needed */}
            </div>
            <div className="sales-data">
                {salesData.length > 0 ? renderSalesData() : <p>Loading...</p>}
            </div>
        </div>
    );
};

export default ReportAndAnalysis;
