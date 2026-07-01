import React, { createContext, useState, useContext } from 'react';

const ResultContext = createContext(null);

export const ResultProvider = ({ children }) => {
  const [result, setResult] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);

  return (
    <ResultContext.Provider value={{ result, setResult, imageSrc, setImageSrc }}>
      {children}
    </ResultContext.Provider>
  );
};

export const useResult = () => useContext(ResultContext);
