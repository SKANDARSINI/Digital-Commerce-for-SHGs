import React from 'react'
import './Sidebar.css'
import {Link} from 'react-router-dom'
import add_product_icon from '../../assets/Product_Cart.svg'
import list_product_icon from '../../assets/Product_list_icon.svg'
import digital_ledger_icon from '../../assets/DigitalLedger.png'

const Sidebar = () => {
  return (
    <div className='sidebar'>
        <Link to={'/addproduct'} style={{textDecoration:"none"}}>
            <div className='sidebar-item'>
                <img src={add_product_icon} alt='' />
                <p>Add Product</p>
            </div>
        </Link>
        <Link to={'/listproduct'} style={{textDecoration:"none"}}>
            <div className='sidebar-item'>
                <img src={list_product_icon} alt='' />
                <p>Product List</p>
            </div>
        </Link>
        <Link to={'/digitalledger'} style={{textDecoration:"none"}}>
            <div className='sidebar-item'>
                <img src={list_product_icon} alt='' />
                <p>Digital Ledger</p>
            </div>
        </Link>
        <Link to={'/reportandanalysis'} style={{textDecoration:"none"}}>
            <div className='sidebar-item'>
                <img src={list_product_icon} alt='' />
                <p>Sales Report</p>
            </div>
        </Link>
    </div>
  )
}

export default Sidebar