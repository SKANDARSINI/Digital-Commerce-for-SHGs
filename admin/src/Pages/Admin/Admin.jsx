import React from 'react'
import './Admin.css'
import Sidebar from '../../Components/Sidebar/Sidebar'
import { Routes,Route } from 'react-router-dom'
import AddProduct from '../../Components/AddProduct/AddProduct'
import ListProduct from '../../Components/ListProduct/ListProduct'
import DigitalLedger from '../../Components/DigitaLledger/DigitalLedger'
import ReportAndAnalysis from '../../Components/ReportAndAnalysis/ReportAndAnalysis'


const Admin = () => {
  return (
    <div className='admin'>
        <Sidebar/>
        <Routes>
            <Route path='/addproduct' element={<AddProduct/>}/>
            <Route path='/listproduct' element={<ListProduct/>}/>
            <Route path='/digitalledger' element={<DigitalLedger/>}/>
            <Route path='/reportandanalysis' element={<ReportAndAnalysis/>}/>
        </Routes>
    </div>
  )
}

export default Admin