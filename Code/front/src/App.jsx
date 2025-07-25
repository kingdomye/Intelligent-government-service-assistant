import { useState } from "react";
import "./App.css";
import { Outlet } from "react-router-dom";

function App() {
  return (
    <>
      <Outlet className="outlet" />
    </>
  );
}

export default App;
