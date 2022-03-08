import { rgbToHex } from "@mui/material";
import React from "react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
// import ReactHowler from "react-howler";
import ReactAudioPlayer from "react-audio-player";

function Sample({ audio_src, data }) {
  console.log(audio_src);
  return (
    <div>
      <table>
        <tr>
          <td>
            <ReactAudioPlayer src={new URL(audio_src)} controls />
          </td>
          <td>
            <BarChart
              width={500}
              height={400}
              data={data}
              layout="vertical"
              margin={{ left: 10 }}
            >
              <Bar dataKey="prob" fill={rgbToHex("#0077CC")}></Bar>
              <CartesianGrid stroke="#ccc"></CartesianGrid>
              <XAxis type="number"></XAxis>
              <YAxis dataKey="class" type="category"></YAxis>
            </BarChart>
          </td>
        </tr>
      </table>
    </div>
  );
}

export default Sample;
