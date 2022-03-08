import React from "react";
import Sample from "./Sample";
import "./Trainer.css";

function Samples({ data }) {
  return (
    <div>
      <h1 className="Title">Some Samples</h1>
      <div>
        {data.map((elem) => (
          <Sample data={elem.data} audio_src={elem.audio_src}></Sample>
        ))}
      </div>
    </div>
  );
}

export default Samples;
