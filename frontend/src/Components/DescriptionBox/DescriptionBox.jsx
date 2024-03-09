import React from 'react'
import './DescriptionBox.css'

const DescriptionBox = () => {
  return (
    <div className='descriptionbox'>
        <div className="descriptionbox-navigator">
            <div className="descriptionbox-nav-box">Description</div>
            <div className="descriptionbox-nav-box fade">Reviews (122)</div>
        </div>
        <div className="descriptionbox-description">
            <p>Handcrafted with care and precision, our Product showcases the skill and artistry of our talented artisans.
                Made from high-quality materials, each piece is unique and embodies the spirit of craftsmanship. Elevate your activity or space with this exquisite creation. 
                <p>Made from the finest materials, our Product Name exudes quality and durability. Whether it's the rich tones of the wood, the vibrant hues of the fabric, or the delicate intricacies of the metalwork, every detail is carefully considered to create a piece that is not only beautiful but also functional. With its timeless appeal and artisanal charm, our Product is sure to become a cherished addition to your home or wardrobe.</p>
                </p>
        </div>
    </div>
  )
}

export default DescriptionBox
