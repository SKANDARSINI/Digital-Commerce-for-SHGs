import React, { useState, useEffect } from 'react';
import './DigitalLedger.css';

const DigitalLedger = () => {
    const [transactions, setTransactions] = useState([]);

    useEffect(() => {
        // Fetch transactions from the server
        fetchTransactions();
    }, []);

    const fetchTransactions = async () => {
        try {
            const response = await fetch('/api/transactions'); // Replace '/api/transactions' with your actual API endpoint
            const data = await response.json();
            setTransactions(data);
        } catch (error) {
            console.error('Error fetching transactions:', error);
        }
    };

    return (
        <div className="digital-ledger">
            <h2>Digital Ledger</h2>
            <table>
                <thead>
                    <tr>
                        <th>Transaction ID</th>
                        <th>Date</th>
                        <th>Amount</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {transactions.map(transaction => (
                        <tr key={transaction.id}>
                            <td>{transaction.id}</td>
                            <td>{transaction.date}</td>
                            <td>{transaction.amount}</td>
                            <td>{transaction.description}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default DigitalLedger;
