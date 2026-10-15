// Inputs: s, i, c, dt, invTauDrag, ugx/ugy/ugz, ux/uy/uz; arithmetic helpers and GPU_OPERATOR_* adapters.

// Exponential particle momentum integration; caller owns its physical eligibility.
const GPU_OPERATOR_REAL xDrag = dt*clampMin(invTauDrag, GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL alpha = exp(-xDrag);
        const GPU_OPERATOR_REAL ax =
            s.gravityX - finiteOr(s.gradPx[c], GPU_OPERATOR_R(0.0))*s.invRhoSolid;
        const GPU_OPERATOR_REAL ay =
            s.gravityY - finiteOr(s.gradPy[c], GPU_OPERATOR_R(0.0))*s.invRhoSolid;
        const GPU_OPERATOR_REAL az =
            s.gravityZ - finiteOr(s.gradPz[c], GPU_OPERATOR_R(0.0))*s.invRhoSolid;

        const GPU_OPERATOR_REAL impulse =
            (xDrag < GPU_OPERATOR_R(1.0e-6))
          ? dt*(GPU_OPERATOR_R(1.0) - GPU_OPERATOR_R(0.5)*xDrag + xDrag*xDrag/GPU_OPERATOR_R(6.0))
          : (GPU_OPERATOR_R(1.0) - alpha)/(invTauDrag + GPU_OPERATOR_TINY(1.0e-300));
    const GPU_OPERATOR_REAL uNewX = ugx + (ux - ugx)*alpha + ax*impulse;
    const GPU_OPERATOR_REAL uNewY = ugy + (uy - ugy)*alpha + ay*impulse;
    const GPU_OPERATOR_REAL uNewZ = ugz + (uz - ugz)*alpha + az*impulse;

        s.pux[i] = uNewX;
        s.puy[i] = uNewY;
        s.puz[i] = uNewZ;
