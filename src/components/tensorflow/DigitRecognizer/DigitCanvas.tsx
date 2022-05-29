import React, { useEffect, useRef, useImperativeHandle } from "react";

import styles from "./DigitRecognizer.module.css";

let drawing = false;

type Props = {
  /** The scale of the canvas (1 is equivalent to 28 x 28 pixels) */
  scale?: number;
};

/** 28 x 28 canvas for user to draw numbers */
export const DigitCanvas = React.forwardRef<HTMLCanvasElement, Props>(
  ({ scale = 1 }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useImperativeHandle(ref, () => canvasRef.current as HTMLCanvasElement);

    const erase = () => {
      const ctx = canvasRef.current?.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, 28, 28);
      }
    };

    useEffect(() => {
      erase();
    }, []);

    const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
      drawing = true;
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (drawing && canvas && ctx) {
        const x = e.clientX - canvas.offsetLeft + window.scrollX;
        const y = e.clientY - canvas.offsetTop + window.scrollY;

        ctx.fillStyle = "#FFF";
        ctx.fillRect(x / scale, y / scale, 2, 2);
      }
    };

    const stopDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
      drawing = false;
    };

    return (
      <div>
        <canvas
          ref={canvasRef}
          id="canvas"
          width={28}
          height={28}
          style={{
            width: `${28 * scale}px`,
            height: `${28 * scale}px`,
          }}
          className={styles.canvas}
          onMouseDown={startDrawing}
          onMouseMove={handleMouseMove}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
        ></canvas>

        <button type="button" onClick={erase}>
          Clear
        </button>
      </div>
    );
  }
);
