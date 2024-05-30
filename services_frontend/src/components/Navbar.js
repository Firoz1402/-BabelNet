import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from './AuthContext'; // Import useAuth hook
import '../styling/Navbar.css';
import '../styling/HomeStyle.css';

const Navbar = () => {
  const { user, logout } = useAuth(); // Access user and logout from AuthContext
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <nav className="navbar">
      <ul className="navbar-nav">
        <li className="nav-item"><Link to="/">HOME</Link></li>
        <li className="nav-item"><Link to="/image-generation">TEXT CLASSIFIER</Link></li>
        <li className="nav-item"><Link to="/image-classification">IMAGE CLASSIFICATION</Link></li>
        {/* <li className="nav-item"><Link to="/register">REGISTER</Link></li> */}
        {user ? (
          <li className="nav-item" onClick={logout}>LOGOUT</li>
        ) : (
          <li className="nav-item"><Link to="/login">LOGIN</Link></li>
        )}
      </ul>

      <div className="navbar-toggle" onClick={toggleMobileMenu}>
        ☰
      </div>

      {isMobileMenuOpen && (
        <ul className="navbar-nav-mobile">
          <li className="nav-item"><Link to="/">HOME</Link></li>
          <li className="nav-item"><Link to="/image-generation">IMAGE GENERATION</Link></li>
          <li className="nav-item"><Link to="/image-classification">IMAGE CLASSIFICATION</Link></li>
          {/* <li className="nav-item"><Link to="/register">REGISTER</Link></li> */}
          {user ? (
            <li className="nav-item" onClick={logout}>LOGOUT</li>
          ) : (
            <li className="nav-item"><Link to="/login">LOGIN</Link></li>
          )}
        </ul>
      )}
    </nav>
  );
}

export default Navbar;

