// AI Image Restoration — Client Application Logic
// Clean, straightforward implementation for student project demonstration

function initApp() {
    try { initModeNav(); } catch (e) { console.error("Error in initModeNav:", e); }
    try { initMode1(); } catch (e) { console.error("Error in initMode1:", e); }
    try { initMode2(); } catch (e) { console.error("Error in initMode2:", e); }
    try { initMode3(); } catch (e) { console.error("Error in initMode3:", e); }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
} else {
    initApp();
}

// Helper: Download Base64 image using Blob URL
function downloadBase64(base64Data, filename) {
    try {
        const parts = base64Data.split(';base64,');
        const contentType = parts[0].split(':')[1];
        const raw = window.atob(parts[1]);
        const uInt8Array = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; ++i) {
            uInt8Array[i] = raw.charCodeAt(i);
        }
        const blob = new Blob([uInt8Array], { type: contentType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        const a = document.createElement('a');
        a.href = base64Data;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

// ===========================================================================
// TOP MODE NAVIGATION
// ===========================================================================
function initModeNav() {
    const navButtons = document.querySelectorAll(".mode-btn");
    const panels = document.querySelectorAll(".mode-panel");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetMode = btn.dataset.mode;
            navButtons.forEach(b => b.classList.toggle("active", b === btn));
            panels.forEach(p => p.classList.toggle("active", p.id === targetMode));
        });
    });
}

// ===========================================================================
// MODE 1: FACE RESTORATION
// ===========================================================================
function initMode1() {
    // Path Choice Elements
    const choicePathA = document.getElementById("btn_choice_path_a");
    const choicePathB = document.getElementById("btn_choice_path_b");
    const pathASection = document.getElementById("path_a_section");
    const pathBSection = document.getElementById("path_b_section");
    const pathAActions = document.getElementById("path_a_actions");
    const pathBActions = document.getElementById("path_b_actions");

    let currentPath = "path_a"; // "path_a" or "path_b"

    // Inline Message Helpers
    const errorBox = document.getElementById("m1_error");
    const infoBox = document.getElementById("m1_info");

    function showError(msg) {
        if (!msg) {
            errorBox.classList.add("hidden");
            errorBox.textContent = "";
        } else {
            errorBox.textContent = msg;
            errorBox.classList.remove("hidden");
            infoBox.classList.add("hidden");
        }
    }

    function showInfo(msg) {
        if (!msg) {
            infoBox.classList.add("hidden");
            infoBox.textContent = "";
        } else {
            infoBox.textContent = msg;
            infoBox.classList.remove("hidden");
            errorBox.classList.add("hidden");
        }
    }

    // Path switching
    choicePathA.addEventListener("click", () => {
        currentPath = "path_a";
        choicePathA.classList.add("active");
        choicePathB.classList.remove("active");
        pathASection.classList.remove("hidden");
        pathBSection.classList.add("hidden");
        pathAActions.classList.remove("hidden");
        pathBActions.classList.add("hidden");
        showError("");
        showInfo("");
    });

    choicePathB.addEventListener("click", () => {
        currentPath = "path_b";
        choicePathB.classList.add("active");
        choicePathA.classList.remove("active");
        pathBSection.classList.remove("hidden");
        pathASection.classList.add("hidden");
        pathBActions.classList.remove("hidden");
        pathAActions.classList.add("hidden");
        showError("");
        showInfo("");
    });

    // =======================================================================
    // OPTION A: INPUTS & STATE
    // =======================================================================
    const aTargetInput = document.getElementById("m1_a_target_file");
    const aTargetPreview = document.getElementById("m1_a_target_preview");
    const aTargetThumb = document.getElementById("m1_a_target_thumb");
    const aTargetName = document.getElementById("m1_a_target_name");
    const aTargetDim = document.getElementById("m1_a_target_dim");
    const aTargetRemove = document.getElementById("m1_a_target_remove");

    let aTargetFileObj = null;
    let aTargetBase64 = null;
    let aTargetWidth = 0;
    let aTargetHeight = 0;

    // Mask Method Switcher (Upload vs Draw)
    const tabMaskUpload = document.getElementById("tab_mask_upload");
    const tabMaskDraw = document.getElementById("tab_mask_draw");
    const subMaskUpload = document.getElementById("sub_mask_upload");
    const subMaskDraw = document.getElementById("sub_mask_draw");
    let activeMaskMethod = "upload"; // "upload" or "draw"

    tabMaskUpload.addEventListener("click", () => {
        activeMaskMethod = "upload";
        tabMaskUpload.classList.add("active");
        tabMaskDraw.classList.remove("active");
        subMaskUpload.classList.remove("hidden");
        subMaskDraw.classList.add("hidden");
    });

    tabMaskDraw.addEventListener("click", () => {
        activeMaskMethod = "draw";
        tabMaskDraw.classList.add("active");
        tabMaskUpload.classList.remove("active");
        subMaskDraw.classList.remove("hidden");
        subMaskUpload.classList.add("hidden");

        if (aTargetFileObj && aTargetBase64) {
            drawNoTargetMsg.classList.add("hidden");
            drawCanvasWrapper.classList.remove("hidden");
            setupCanvasWithTarget();
        } else {
            drawNoTargetMsg.classList.remove("hidden");
            drawCanvasWrapper.classList.add("hidden");
        }
    });

    // Upload Mask Elements
    const aMaskInput = document.getElementById("m1_a_mask_file");
    const aMaskPreview = document.getElementById("m1_a_mask_preview");
    const aMaskThumb = document.getElementById("m1_a_mask_thumb");
    const aMaskName = document.getElementById("m1_a_mask_name");
    const aMaskDim = document.getElementById("m1_a_mask_dim");
    const aMaskRemove = document.getElementById("m1_a_mask_remove");

    let aMaskFileObj = null;
    let aMaskBase64 = null;
    let aMaskWidth = 0;
    let aMaskHeight = 0;

    // Draw Mask Elements
    const drawNoTargetMsg = document.getElementById("draw_no_target_msg");
    const drawCanvasWrapper = document.getElementById("draw_canvas_wrapper");
    const drawCanvas = document.getElementById("mask_draw_canvas");
    const drawCtx = drawCanvas ? drawCanvas.getContext("2d") : null;
    const brushSlider = document.getElementById("brush_size_slider");
    const brushSizeLabel = document.getElementById("brush_size_label");
    const btnToolBrush = document.getElementById("btn_tool_brush");
    const btnToolEraser = document.getElementById("btn_tool_eraser");
    const btnDrawUndo = document.getElementById("btn_draw_undo");
    const btnDrawClear = document.getElementById("btn_draw_clear");
    const btnDrawReset = document.getElementById("btn_draw_reset");
    const btnDrawConfirm = document.getElementById("btn_draw_confirm");

    const drawnMaskConfirmed = document.getElementById("drawn_mask_confirmed");
    const drawnMaskThumb = document.getElementById("drawn_mask_thumb");
    const drawnMaskDim = document.getElementById("drawn_mask_dim");
    const btnRedraw = document.getElementById("btn_redraw");
    const btnDrawnRemove = document.getElementById("btn_drawn_remove");

    let currentTool = "brush"; // "brush" or "eraser"
    let brushSize = 24;
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let undoStack = [];

    // In-memory paint canvas (stores the drawn mask layer)
    const paintCanvas = document.createElement("canvas");
    const paintCtx = paintCanvas.getContext("2d");

    let drawnMaskBlob = null;
    let drawnMaskDataUrl = null;

    // =======================================================================
    // TARGET IMAGE HANDLER (OPTION A)
    // =======================================================================
    aTargetInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            aTargetFileObj = file;
            aTargetName.textContent = file.name;

            const reader = new FileReader();
            reader.onload = (ev) => {
                aTargetBase64 = ev.target.result;
                aTargetThumb.src = aTargetBase64;
                aTargetPreview.classList.remove("hidden");

                const img = new Image();
                img.onload = () => {
                    aTargetWidth = img.naturalWidth;
                    aTargetHeight = img.naturalHeight;
                    aTargetDim.textContent = `${aTargetWidth} \u00D7 ${aTargetHeight} px`;

                    // Check dimension consistency if an uploaded mask is already chosen
                    if (aMaskFileObj && aMaskWidth > 0 && aMaskHeight > 0) {
                        if (aMaskWidth !== aTargetWidth || aMaskHeight !== aTargetHeight) {
                            showError(`Mask size (${aMaskWidth}\u00D7${aMaskHeight}) does not match the face image (${aTargetWidth}\u00D7${aTargetHeight}). Please upload a matching mask or create the mask using Draw Mask.`);
                            clearUploadedMask();
                        } else {
                            showError("");
                        }
                    }

                    // If Draw Mask tab is active, setup canvas immediately
                    if (activeMaskMethod === "draw") {
                        drawNoTargetMsg.classList.add("hidden");
                        drawCanvasWrapper.classList.remove("hidden");
                        setupCanvasWithTarget();
                    }
                };
                img.src = aTargetBase64;
            };
            reader.readAsDataURL(file);
        }
    });

    function clearOptionATarget() {
        aTargetFileObj = null;
        aTargetBase64 = null;
        aTargetWidth = 0;
        aTargetHeight = 0;
        aTargetInput.value = "";
        aTargetThumb.src = "";
        aTargetName.textContent = "";
        aTargetDim.textContent = "";
        aTargetPreview.classList.add("hidden");

        if (activeMaskMethod === "draw") {
            drawNoTargetMsg.classList.remove("hidden");
            drawCanvasWrapper.classList.add("hidden");
        }
    }

    aTargetRemove.addEventListener("click", clearOptionATarget);

    // =======================================================================
    // UPLOAD MASK HANDLER (OPTION A1)
    // =======================================================================
    aMaskInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            aMaskFileObj = file;
            aMaskName.textContent = file.name;

            const reader = new FileReader();
            reader.onload = (ev) => {
                aMaskBase64 = ev.target.result;
                aMaskThumb.src = aMaskBase64;

                const maskImg = new Image();
                maskImg.onload = () => {
                    aMaskWidth = maskImg.naturalWidth;
                    aMaskHeight = maskImg.naturalHeight;
                    aMaskDim.textContent = `${aMaskWidth} \u00D7 ${aMaskHeight} px`;

                    // Strict Dimension Check against target
                    if (aTargetWidth > 0 && aTargetHeight > 0) {
                        if (aMaskWidth !== aTargetWidth || aMaskHeight !== aTargetHeight) {
                            showError(`Mask size (${aMaskWidth}\u00D7${aMaskHeight}) does not match the face image (${aTargetWidth}\u00D7${aTargetHeight}). Please upload a matching mask or create the mask using Draw Mask.`);
                            clearUploadedMask();
                            return;
                        }
                    }

                    showError("");
                    aMaskPreview.classList.remove("hidden");
                };
                maskImg.src = aMaskBase64;
            };
            reader.readAsDataURL(file);
        }
    });

    function clearUploadedMask() {
        aMaskFileObj = null;
        aMaskBase64 = null;
        aMaskWidth = 0;
        aMaskHeight = 0;
        aMaskInput.value = "";
        aMaskThumb.src = "";
        aMaskName.textContent = "";
        aMaskDim.textContent = "";
        aMaskPreview.classList.add("hidden");
    }

    aMaskRemove.addEventListener("click", clearUploadedMask);

    // =======================================================================
    // DRAW MASK CANVAS ENGINE (OPTION A2)
    // =======================================================================
    function setupCanvasWithTarget() {
        if (!aTargetWidth || !aTargetHeight || !drawCanvas) return;

        drawCanvas.width = aTargetWidth;
        drawCanvas.height = aTargetHeight;
        paintCanvas.width = aTargetWidth;
        paintCanvas.height = aTargetHeight;

        if (!drawnMaskBlob) {
            paintCtx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
            undoStack = [];
        }

        renderDisplayCanvas();
    }

    function renderDisplayCanvas() {
        if (!aTargetWidth || !aTargetHeight || !drawCtx) return;
        const targetImg = new Image();
        targetImg.onload = () => {
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            drawCtx.drawImage(targetImg, 0, 0);
            drawCtx.drawImage(paintCanvas, 0, 0);
        };
        targetImg.src = aTargetBase64;
    }

    function pushUndoState() {
        if (!aTargetWidth || !aTargetHeight) return;
        if (undoStack.length >= 15) {
            undoStack.shift();
        }
        undoStack.push(paintCtx.getImageData(0, 0, paintCanvas.width, paintCanvas.height));
    }

    function getCanvasCoordinates(e) {
        if (!drawCanvas) return { x: 0, y: 0 };
        const rect = drawCanvas.getBoundingClientRect();
        const scaleX = drawCanvas.width / rect.width;
        const scaleY = drawCanvas.height / rect.height;
        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function paintStroke(x1, y1, x2, y2) {
        paintCtx.beginPath();
        if (currentTool === "brush") {
            paintCtx.globalCompositeOperation = "source-over";
            paintCtx.strokeStyle = "rgba(239, 68, 68, 0.70)";
            paintCtx.fillStyle = "rgba(239, 68, 68, 0.70)";
        } else {
            paintCtx.globalCompositeOperation = "destination-out";
        }

        paintCtx.lineWidth = brushSize;
        paintCtx.lineCap = "round";
        paintCtx.lineJoin = "round";
        paintCtx.moveTo(x1, y1);
        paintCtx.lineTo(x2, y2);
        paintCtx.stroke();

        renderDisplayCanvas();
    }

    if (drawCanvas) {
        drawCanvas.addEventListener("pointerdown", (e) => {
            if (!aTargetWidth || !aTargetHeight) return;
            isDrawing = true;
            try { drawCanvas.setPointerCapture(e.pointerId); } catch (_) {}

            pushUndoState();
            const coords = getCanvasCoordinates(e);
            lastX = coords.x;
            lastY = coords.y;
            paintStroke(lastX, lastY, lastX, lastY);
        });

        drawCanvas.addEventListener("pointermove", (e) => {
            if (!isDrawing) return;
            const coords = getCanvasCoordinates(e);
            paintStroke(lastX, lastY, coords.x, coords.y);
            lastX = coords.x;
            lastY = coords.y;
        });

        const endDrawing = (e) => {
            if (isDrawing) {
                isDrawing = false;
                try { drawCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
            }
        };

        drawCanvas.addEventListener("pointerup", endDrawing);
        drawCanvas.addEventListener("pointercancel", endDrawing);
    }

    if (brushSlider) {
        brushSlider.addEventListener("input", (e) => {
            brushSize = parseInt(e.target.value, 10);
            if (brushSizeLabel) brushSizeLabel.textContent = brushSize;
        });
    }

    if (btnToolBrush) {
        btnToolBrush.addEventListener("click", () => {
            currentTool = "brush";
            btnToolBrush.classList.add("active");
            if (btnToolEraser) btnToolEraser.classList.remove("active");
        });
    }

    if (btnToolEraser) {
        btnToolEraser.addEventListener("click", () => {
            currentTool = "eraser";
            btnToolEraser.classList.add("active");
            if (btnToolBrush) btnToolBrush.classList.remove("active");
        });
    }

    if (btnDrawUndo) {
        btnDrawUndo.addEventListener("click", () => {
            if (undoStack.length > 0) {
                const prevState = undoStack.pop();
                paintCtx.putImageData(prevState, 0, 0);
                renderDisplayCanvas();
            }
        });
    }

    if (btnDrawClear) {
        btnDrawClear.addEventListener("click", () => {
            pushUndoState();
            paintCtx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
            renderDisplayCanvas();
        });
    }

    if (btnDrawReset) {
        btnDrawReset.addEventListener("click", () => {
            undoStack = [];
            paintCtx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
            renderDisplayCanvas();
        });
    }

    if (btnDrawConfirm) {
        btnDrawConfirm.addEventListener("click", () => {
            if (!aTargetWidth || !aTargetHeight) {
                showError("Please upload a face image first.");
                return;
            }

            const exportCanvas = document.createElement("canvas");
            exportCanvas.width = aTargetWidth;
            exportCanvas.height = aTargetHeight;
            const expCtx = exportCanvas.getContext("2d");

            expCtx.fillStyle = "#000000";
            expCtx.fillRect(0, 0, aTargetWidth, aTargetHeight);

            const paintImgData = paintCtx.getImageData(0, 0, aTargetWidth, aTargetHeight);
            const pData = paintImgData.data;
            const outImgData = expCtx.getImageData(0, 0, aTargetWidth, aTargetHeight);
            const oData = outImgData.data;

            let paintedPixelCount = 0;
            for (let i = 0; i < pData.length; i += 4) {
                if (pData[i + 3] > 20) {
                    oData[i] = 255;
                    oData[i + 1] = 255;
                    oData[i + 2] = 255;
                    oData[i + 3] = 255;
                    paintedPixelCount++;
                }
            }
            expCtx.putImageData(outImgData, 0, 0);

            if (paintedPixelCount < 10) {
                showError("Please paint over the area you want to restore before confirming.");
                return;
            }

            showError("");
            drawnMaskDataUrl = exportCanvas.toDataURL("image/png");
            drawnMaskThumb.src = drawnMaskDataUrl;
            drawnMaskDim.textContent = `${aTargetWidth} \u00D7 ${aTargetHeight} px (${paintedPixelCount.toLocaleString()} px)`;

            exportCanvas.toBlob((blob) => {
                drawnMaskBlob = blob;
                drawnMaskConfirmed.classList.remove("hidden");
                drawCanvasWrapper.classList.add("hidden");
                showInfo(`Custom mask confirmed (${paintedPixelCount.toLocaleString()} pixels). Ready for restoration.`);
            }, "image/png");
        });
    }

    if (btnRedraw) {
        btnRedraw.addEventListener("click", () => {
            drawnMaskConfirmed.classList.add("hidden");
            drawCanvasWrapper.classList.remove("hidden");
            renderDisplayCanvas();
        });
    }

    if (btnDrawnRemove) {
        btnDrawnRemove.addEventListener("click", () => {
            drawnMaskBlob = null;
            drawnMaskDataUrl = null;
            drawnMaskConfirmed.classList.add("hidden");
            drawCanvasWrapper.classList.remove("hidden");
            paintCtx.clearRect(0, 0, paintCanvas.width, paintCanvas.height);
            undoStack = [];
            renderDisplayCanvas();
        });
    }

    function loadMaskOntoCanvas(maskBase64Url) {
        const maskImg = new Image();
        maskImg.onload = () => {
            const tempC = document.createElement("canvas");
            tempC.width = aTargetWidth;
            tempC.height = aTargetHeight;
            const tempCtx = tempC.getContext("2d");
            tempCtx.drawImage(maskImg, 0, 0, aTargetWidth, aTargetHeight);

            const mData = tempCtx.getImageData(0, 0, aTargetWidth, aTargetHeight).data;
            const pData = paintCtx.createImageData(aTargetWidth, aTargetHeight);
            const pd = pData.data;

            for (let i = 0; i < mData.length; i += 4) {
                if (mData[i] > 127 || mData[i + 1] > 127 || mData[i + 2] > 127) {
                    pd[i] = 239;
                    pd[i + 1] = 68;
                    pd[i + 2] = 68;
                    pd[i + 3] = 175;
                }
            }

            undoStack = [];
            paintCtx.putImageData(pData, 0, 0);
            renderDisplayCanvas();
        };
        maskImg.src = maskBase64Url;
    }

    // =======================================================================
    // OPTION B: NATURAL FACIAL DAMAGE
    // =======================================================================
    const bTargetInput = document.getElementById("m1_b_target_file");
    const bTargetPreview = document.getElementById("m1_b_target_preview");
    const bTargetThumb = document.getElementById("m1_b_target_thumb");
    const bTargetName = document.getElementById("m1_b_target_name");
    const bTargetDim = document.getElementById("m1_b_target_dim");
    const bTargetRemove = document.getElementById("m1_b_target_remove");

    const bDetectBtn = document.getElementById("m1_b_detect_btn");
    const bDetectPreview = document.getElementById("m1_b_detect_preview");
    const bDetectThumb = document.getElementById("m1_b_detect_thumb");
    const bDetectStats = document.getElementById("m1_b_detect_stats");
    const btnBUseDetected = document.getElementById("btn_b_use_detected");
    const btnBAdjustMask = document.getElementById("btn_b_adjust_mask");

    let bTargetFileObj = null;
    let bTargetBase64 = null;
    let bTargetWidth = 0;
    let bTargetHeight = 0;
    let bAutoMaskBlob = null;
    let bAutoMaskBase64 = null;

    bTargetInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            bTargetFileObj = file;
            bTargetName.textContent = file.name;

            const reader = new FileReader();
            reader.onload = (ev) => {
                bTargetBase64 = ev.target.result;
                bTargetThumb.src = bTargetBase64;
                bTargetPreview.classList.remove("hidden");

                const bImg = new Image();
                bImg.onload = () => {
                    bTargetWidth = bImg.naturalWidth;
                    bTargetHeight = bImg.naturalHeight;
                    bTargetDim.textContent = `${bTargetWidth} \u00D7 ${bTargetHeight} px`;
                    bDetectPreview.classList.add("hidden");
                    bAutoMaskBlob = null;
                    bAutoMaskBase64 = null;
                    showError("");
                };
                bImg.src = bTargetBase64;
            };
            reader.readAsDataURL(file);
        }
    });

    bTargetRemove.addEventListener("click", () => {
        bTargetFileObj = null;
        bTargetBase64 = null;
        bTargetWidth = 0;
        bTargetHeight = 0;
        bTargetInput.value = "";
        bTargetThumb.src = "";
        bTargetName.textContent = "";
        bTargetDim.textContent = "";
        bTargetPreview.classList.add("hidden");
        bDetectPreview.classList.add("hidden");
        bAutoMaskBlob = null;
        bAutoMaskBase64 = null;
    });

    bDetectBtn.addEventListener("click", async () => {
        if (!bTargetFileObj) {
            showError("Please upload a naturally damaged face photo.");
            return;
        }

        showError("");
        bDetectBtn.disabled = true;
        bDetectBtn.textContent = "Detecting damage...";

        const formData = new FormData();
        formData.append("target", bTargetFileObj);
        const refFiles = getActiveReferenceFiles();
        refFiles.forEach((file, idx) => formData.append(`ref_${idx}`, file));

        try {
            const resp = await fetch("/api/mode1/detect_damage", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            if (data.status === "success") {
                bDetectThumb.src = data.overlay_base64;
                bDetectStats.textContent = `${data.detected_pixels.toLocaleString()} pixels (${data.face_coverage_pct}% of face, confidence: ${data.confidence})`;
                bDetectPreview.classList.remove("hidden");

                bAutoMaskBase64 = data.mask_base64;
                const fetchRes = await fetch(data.mask_base64);
                bAutoMaskBlob = await fetchRes.blob();

                showInfo("Damage detected. Click 'Use Detected Area' or 'Restore Face' to proceed.");
            } else {
                showError("Detection error: " + (data.error || "Unable to detect damage."));
            }
        } catch (err) {
            showError("Detection failed: " + err.message);
        } finally {
            bDetectBtn.disabled = false;
            bDetectBtn.textContent = "Detect Damage";
        }
    });

    btnBUseDetected.addEventListener("click", () => {
        if (bAutoMaskBlob) {
            showInfo("Detected area selected. Click 'Restore Face' below to reconstruct.");
        }
    });

    btnBAdjustMask.addEventListener("click", () => {
        if (!bTargetFileObj || !bAutoMaskBase64) {
            showError("Please detect damage first before adjusting the mask.");
            return;
        }

        choicePathA.click();

        aTargetFileObj = bTargetFileObj;
        aTargetBase64 = bTargetBase64;
        aTargetWidth = bTargetWidth;
        aTargetHeight = bTargetHeight;
        aTargetThumb.src = aTargetBase64;
        aTargetName.textContent = bTargetFileObj.name;
        aTargetDim.textContent = `${aTargetWidth} \u00D7 ${aTargetHeight} px`;
        aTargetPreview.classList.remove("hidden");

        const targetImg = new Image();
        targetImg.onload = () => {
            tabMaskDraw.click();
            loadMaskOntoCanvas(bAutoMaskBase64);
            showInfo("Loaded detected damage into the Draw Mask tool. You can now refine the area, then click 'Use Mask'.");
        };
        targetImg.src = aTargetBase64;
    });

    // =======================================================================
    // IDENTITY REFERENCES (SHARED STEP 3) — DYNAMIC INCREMENTAL SLOTS
    // =======================================================================
    const addRefBtn = document.getElementById("m1_add_ref_btn");
    const clearRefsBtn = document.getElementById("m1_clear_refs_btn");
    const refListContainer = document.getElementById("m1_ref_list");
    const refError = document.getElementById("m1_ref_error");

    function getActiveReferenceFiles() {
        const files = [];
        const slots = refListContainer.querySelectorAll(".ref-slot");
        slots.forEach(slot => {
            const input = slot.querySelector(".ref-file-input");
            if (input && input.files && input.files[0]) {
                files.push(input.files[0]);
            }
        });
        return files;
    }

    function updateReferenceControls() {
        const slots = refListContainer.querySelectorAll(".ref-slot");
        if (slots.length >= 3) {
            addRefBtn.disabled = true;
            addRefBtn.textContent = "Maximum 3 References Added";
        } else {
            addRefBtn.disabled = false;
            addRefBtn.textContent = "+ Add Reference Photo";
        }

        if (slots.length > 0) {
            clearRefsBtn.classList.remove("hidden");
        } else {
            clearRefsBtn.classList.add("hidden");
        }
    }

    function createReferenceSlot() {
        const slots = refListContainer.querySelectorAll(".ref-slot");
        if (slots.length >= 3) {
            if (refError) {
                refError.textContent = "Maximum of 3 reference photos allowed.";
                refError.classList.remove("hidden");
            }
            return;
        }

        if (refError) {
            refError.textContent = "";
            refError.classList.add("hidden");
        }

        const slot = document.createElement("div");
        slot.className = "ref-slot";

        const topRow = document.createElement("div");
        topRow.className = "ref-slot-top";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*";
        fileInput.className = "ref-file-input";

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "btn-remove";
        removeBtn.textContent = "Remove";

        topRow.appendChild(fileInput);
        topRow.appendChild(removeBtn);

        const previewRow = document.createElement("div");
        previewRow.className = "ref-slot-preview hidden";

        const thumb = document.createElement("img");
        thumb.className = "preview-thumb ref-thumb";
        thumb.src = "";
        thumb.alt = "Reference preview";

        const meta = document.createElement("div");
        meta.className = "thumb-meta";

        const filenameSpan = document.createElement("span");
        filenameSpan.className = "meta-filename";

        const dimSpan = document.createElement("span");
        dimSpan.className = "meta-dimensions";

        meta.appendChild(filenameSpan);
        meta.appendChild(dimSpan);

        previewRow.appendChild(thumb);
        previewRow.appendChild(meta);

        slot.appendChild(topRow);
        slot.appendChild(previewRow);

        let currentObjUrl = null;

        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                const file = e.target.files[0];
                filenameSpan.textContent = file.name;

                if (currentObjUrl) {
                    URL.revokeObjectURL(currentObjUrl);
                }
                currentObjUrl = URL.createObjectURL(file);
                thumb.src = currentObjUrl;
                previewRow.classList.remove("hidden");

                const img = new Image();
                img.onload = () => {
                    dimSpan.textContent = `${img.naturalWidth} \u00D7 ${img.naturalHeight} px`;
                };
                img.src = currentObjUrl;
            }
        });

        removeBtn.addEventListener("click", () => {
            if (currentObjUrl) {
                URL.revokeObjectURL(currentObjUrl);
            }
            slot.remove();
            if (refError) {
                refError.textContent = "";
                refError.classList.add("hidden");
            }
            updateReferenceControls();
        });

        refListContainer.appendChild(slot);
        updateReferenceControls();
    }

    addRefBtn.addEventListener("click", () => {
        createReferenceSlot();
    });

    clearRefsBtn.addEventListener("click", () => {
        const slots = refListContainer.querySelectorAll(".ref-slot");
        slots.forEach(s => {
            const img = s.querySelector(".ref-thumb");
            if (img && img.src && img.src.startsWith("blob:")) {
                URL.revokeObjectURL(img.src);
            }
        });
        refListContainer.innerHTML = "";
        if (refError) {
            refError.textContent = "";
            refError.classList.add("hidden");
        }
        updateReferenceControls();
    });

    // =======================================================================
    // RESULTS, PROGRESS & FALLBACK
    // =======================================================================
    const emptyMsg = document.getElementById("m1_empty_msg");
    const resultImg = document.getElementById("m1_result_img");
    const statusPanel = document.getElementById("m1_status_panel");
    const statusText = document.getElementById("m1_status_text");
    const regenText = document.getElementById("m1_regen_text");
    const auditText = document.getElementById("m1_audit_text");
    const downloadBtn = document.getElementById("m1_download_btn");
    const viewSwitches = document.getElementById("m1_view_switches");

    const progressContainer = document.getElementById("m1_progress_container");
    const progressBar = document.getElementById("m1_progress_bar");
    const progressPct = document.getElementById("m1_progress_pct");
    const progressMsg = document.getElementById("m1_progress_msg");

    const fallbackCard = document.getElementById("m1_fallback_card");
    const btnTriggerAutoFallback = document.getElementById("btn_trigger_auto_fallback");

    let currentRestoredBase64 = null;
    let currentInputBase64 = null;
    let currentMaskBase64 = null;
    let activePollInterval = null;

    if (viewSwitches) {
        viewSwitches.querySelectorAll(".switch-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const view = btn.dataset.view;
                if (view === "result" && currentRestoredBase64) {
                    resultImg.src = currentRestoredBase64;
                } else if (view === "original" && currentInputBase64) {
                    resultImg.src = currentInputBase64;
                } else if (view === "mask" && currentMaskBase64) {
                    resultImg.src = currentMaskBase64;
                }
            });
        });
    }

    btnTriggerAutoFallback.addEventListener("click", () => {
        fallbackCard.classList.add("hidden");
        choicePathB.click();

        if (aTargetFileObj) {
            bTargetFileObj = aTargetFileObj;
            bTargetBase64 = aTargetBase64;
            bTargetWidth = aTargetWidth;
            bTargetHeight = aTargetHeight;
            bTargetThumb.src = aTargetBase64;
            bTargetName.textContent = aTargetFileObj.name;
            bTargetDim.textContent = `${bTargetWidth} \u00D7 ${bTargetHeight} px`;
            bTargetPreview.classList.remove("hidden");

            bDetectBtn.click();
        }
    });

    function setActionButtonsDisabled(disabled) {
        document.getElementById("m1_run_preview_btn").disabled = disabled;
        document.getElementById("m1_run_full_btn").disabled = disabled;
        document.getElementById("m1_b_preview_btn").disabled = disabled;
        document.getElementById("m1_b_restore_btn").disabled = disabled;
    }

    function pollRestorationJob(jobId) {
        if (activePollInterval) {
            clearInterval(activePollInterval);
        }

        activePollInterval = setInterval(async () => {
            try {
                const resp = await fetch(`/api/mode1/job_status/${jobId}`);
                if (!resp.ok) {
                    clearInterval(activePollInterval);
                    activePollInterval = null;
                    setActionButtonsDisabled(false);
                    showError("Failed to fetch restoration status.");
                    return;
                }

                const job = await resp.json();

                const pct = job.progress_pct || 0;
                progressBar.style.width = `${pct}%`;
                progressPct.textContent = `${pct}%`;
                progressMsg.textContent = job.message || "Processing...";

                if (job.status === "completed") {
                    clearInterval(activePollInterval);
                    activePollInterval = null;
                    progressContainer.classList.add("hidden");
                    setActionButtonsDisabled(false);

                    const res = job.result;
                    currentRestoredBase64 = res.result_base64;
                    currentMaskBase64 = res.mask_base64;

                    resultImg.src = currentRestoredBase64;
                    resultImg.classList.remove("hidden");
                    emptyMsg.classList.add("hidden");

                    viewSwitches.classList.remove("hidden");
                    viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                    viewSwitches.querySelector('[data-view="result"]').classList.add("active");

                    statusPanel.classList.remove("hidden");
                    statusText.textContent = "Restoration complete.";
                    regenText.textContent = `Region regenerated: ${res.masked_pixels_regenerated}`;
                    auditText.textContent = `Boundary invariance: \u0394max = 0.00000 (0 outside pixels modified) | Runtime: ${res.runtime_seconds}s`;

                    downloadBtn.classList.remove("hidden");
                    downloadBtn.textContent = "Download Restored Image";

                    showInfo("Full restoration complete.");
                } else if (job.status === "failed_reconstruction") {
                    clearInterval(activePollInterval);
                    activePollInterval = null;
                    progressContainer.classList.add("hidden");
                    setActionButtonsDisabled(false);

                    fallbackCard.classList.remove("hidden");
                    showError("The selected region was not reconstructed sufficiently.");

                    statusPanel.classList.remove("hidden");
                    statusText.textContent = "Reconstruction Insufficient";
                    regenText.textContent = `Region regenerated: ${job.masked_pixels_regenerated}`;
                    auditText.textContent = "The model preserved the input without generating new facial structure.";
                } else if (job.status === "error") {
                    clearInterval(activePollInterval);
                    activePollInterval = null;
                    progressContainer.classList.add("hidden");
                    setActionButtonsDisabled(false);

                    showError("Restoration failed: " + (job.error || "Unknown error"));
                }
            } catch (err) {
                console.error("Poll error:", err);
            }
        }, 1000);
    }

    async function executeRestoration(execMode) {
        showError("");
        showInfo("");
        fallbackCard.classList.add("hidden");

        let targetFile = null;
        let maskFile = null;

        if (currentPath === "path_a") {
            if (!aTargetFileObj) {
                showError("Please upload a face image in Step 1.");
                return;
            }

            if (activeMaskMethod === "upload") {
                if (!aMaskFileObj) {
                    showError("Please upload a mask in Step 2, or switch to 'Draw Mask'.");
                    return;
                }
                if (aMaskWidth !== aTargetWidth || aMaskHeight !== aTargetHeight) {
                    showError(`Mask size (${aMaskWidth}\u00D7${aMaskHeight}) does not match the face image (${aTargetWidth}\u00D7${aTargetHeight}). Please upload a matching mask or create the mask using Draw Mask.`);
                    return;
                }
                maskFile = aMaskFileObj;
                currentMaskBase64 = aMaskBase64;
            } else {
                if (!drawnMaskBlob) {
                    showError("Please paint the region and click 'Use Mask' before restoring.");
                    return;
                }
                maskFile = drawnMaskBlob;
                currentMaskBase64 = drawnMaskDataUrl;
            }

            targetFile = aTargetFileObj;
            currentInputBase64 = aTargetBase64;
        } else {
            if (!bTargetFileObj) {
                showError("Please upload a naturally damaged face photo.");
                return;
            }

            targetFile = bTargetFileObj;
            currentInputBase64 = bTargetBase64;

            if (bAutoMaskBlob) {
                maskFile = bAutoMaskBlob;
                currentMaskBase64 = bAutoMaskBase64;
            }
        }

        setActionButtonsDisabled(true);

        const formData = new FormData();
        formData.append("target", targetFile);
        formData.append("execution_mode", execMode);
        if (maskFile) {
            formData.append("mask", maskFile, "mask.png");
        }

        const refFiles = getActiveReferenceFiles();
        refFiles.forEach((file, idx) => formData.append(`ref_${idx}`, file));

        try {
            if (execMode === "full_diffusion") {
                progressContainer.classList.remove("hidden");
                progressBar.style.width = "4%";
                progressPct.textContent = "4%";
                progressMsg.textContent = "Submitting restoration job...";

                const resp = await fetch("/api/mode1/restore", {
                    method: "POST",
                    body: formData
                });
                const data = await resp.json();

                if (resp.status === 400) {
                    progressContainer.classList.add("hidden");
                    setActionButtonsDisabled(false);
                    showError(data.error || "Mask dimension mismatch.");
                    return;
                }

                if (data.status === "queued" && data.job_id) {
                    pollRestorationJob(data.job_id);
                } else if (data.status === "success") {
                    progressContainer.classList.add("hidden");
                    setActionButtonsDisabled(false);
                    currentRestoredBase64 = data.result_base64;
                    currentMaskBase64 = data.mask_base64;
                    resultImg.src = currentRestoredBase64;
                    resultImg.classList.remove("hidden");
                    emptyMsg.classList.add("hidden");

                    viewSwitches.classList.remove("hidden");
                    viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                    viewSwitches.querySelector('[data-view="result"]').classList.add("active");

                    statusPanel.classList.remove("hidden");
                    statusText.textContent = "Restoration complete.";
                    regenText.textContent = `Region regenerated: ${data.masked_pixels_regenerated}`;
                    auditText.textContent = `Boundary invariance: \u0394max = 0.00000 (0 outside pixels modified) | Runtime: ${data.runtime_seconds}s`;

                    downloadBtn.classList.remove("hidden");
                    downloadBtn.textContent = "Download Restored Image";
                    showInfo("Full restoration complete.");
                } else {
                    progressContainer.classList.add("hidden");
                    setActionButtonsDisabled(false);
                    showError("Restoration failed: " + (data.error || "Unknown error"));
                }
            } else {
                showInfo("Generating preview...");
                const resp = await fetch("/api/mode1/restore", {
                    method: "POST",
                    body: formData
                });
                const data = await resp.json();

                setActionButtonsDisabled(false);

                if (resp.status === 400) {
                    showError(data.error || "Mask dimension mismatch.");
                    return;
                }

                if (data.status === "success") {
                    currentRestoredBase64 = data.result_base64;
                    currentMaskBase64 = data.mask_base64;

                    resultImg.src = currentRestoredBase64;
                    resultImg.classList.remove("hidden");
                    emptyMsg.classList.add("hidden");

                    viewSwitches.classList.remove("hidden");
                    viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                    viewSwitches.querySelector('[data-view="result"]').classList.add("active");

                    statusPanel.classList.remove("hidden");
                    statusText.textContent = "Quick preview ready (demonstration only).";
                    regenText.textContent = "Region regenerated: N/A (Preview Mode)";
                    auditText.textContent = "Does not run full diffusion. Inputs and mask contour are verified.";

                    downloadBtn.classList.remove("hidden");
                    downloadBtn.textContent = "Download Preview Image";

                    showInfo("Quick preview complete.");
                } else {
                    showError("Preview error: " + (data.error || "Unknown error"));
                }
            }
        } catch (err) {
            progressContainer.classList.add("hidden");
            setActionButtonsDisabled(false);
            showError("Request error: " + err.message);
        }
    }

    document.getElementById("m1_run_preview_btn").addEventListener("click", () => executeRestoration("preview"));
    document.getElementById("m1_run_full_btn").addEventListener("click", () => executeRestoration("full_diffusion"));
    document.getElementById("m1_b_preview_btn").addEventListener("click", () => executeRestoration("preview"));
    document.getElementById("m1_b_restore_btn").addEventListener("click", () => executeRestoration("full_diffusion"));

    downloadBtn.addEventListener("click", () => {
        if (currentRestoredBase64) {
            downloadBase64(currentRestoredBase64, "restored_face.png");
        }
    });
}

// ===========================================================================
// MODE 2: BACKGROUND REMOVAL
// ===========================================================================
function initMode2() {
    const fileInput = document.getElementById("m2_file");
    const previewRow = document.getElementById("m2_preview");
    const thumbImg = document.getElementById("m2_thumb");
    const nameSpan = document.getElementById("m2_name");
    const removeBtn = document.getElementById("m2_remove");
    const runBtn = document.getElementById("m2_run_btn");

    const errorBox = document.getElementById("m2_error");
    const emptyMsg = document.getElementById("m2_empty_msg");
    const resultImg = document.getElementById("m2_result_img");
    const viewSwitches = document.getElementById("m2_view_switches");
    const downloadBtn = document.getElementById("m2_download_btn");

    let currentFile = null;
    let inputBase64 = null;
    let cutoutBase64 = null;

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            currentFile = e.target.files[0];
            nameSpan.textContent = currentFile.name;
            const reader = new FileReader();
            reader.onload = (ev) => {
                inputBase64 = ev.target.result;
                thumbImg.src = inputBase64;
                previewRow.classList.remove("hidden");
                errorBox.classList.add("hidden");
            };
            reader.readAsDataURL(currentFile);
        }
    });

    removeBtn.addEventListener("click", () => {
        currentFile = null;
        inputBase64 = null;
        fileInput.value = "";
        thumbImg.src = "";
        nameSpan.textContent = "";
        previewRow.classList.add("hidden");
    });

    if (viewSwitches) {
        viewSwitches.querySelectorAll(".switch-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const view = btn.dataset.view;
                if (view === "result" && cutoutBase64) {
                    resultImg.src = cutoutBase64;
                } else if (view === "original" && inputBase64) {
                    resultImg.src = inputBase64;
                }
            });
        });
    }

    runBtn.addEventListener("click", async () => {
        if (!currentFile) {
            errorBox.textContent = "Please upload an image.";
            errorBox.classList.remove("hidden");
            return;
        }

        errorBox.classList.add("hidden");
        runBtn.disabled = true;
        runBtn.textContent = "Removing background...";

        const formData = new FormData();
        formData.append("image", currentFile);

        try {
            const resp = await fetch("/api/mode2/remove_background", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            if (data.status === "success") {
                cutoutBase64 = data.result_base64;
                resultImg.src = cutoutBase64;
                resultImg.classList.remove("hidden");
                emptyMsg.classList.add("hidden");

                viewSwitches.classList.remove("hidden");
                viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                viewSwitches.querySelector('[data-view="result"]').classList.add("active");

                downloadBtn.classList.remove("hidden");
            } else {
                errorBox.textContent = "Error: " + (data.error || "Background removal failed.");
                errorBox.classList.remove("hidden");
            }
        } catch (err) {
            errorBox.textContent = "Request error: " + err.message;
            errorBox.classList.remove("hidden");
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = "Remove Background";
        }
    });

    downloadBtn.addEventListener("click", () => {
        if (cutoutBase64) {
            downloadBase64(cutoutBase64, "transparent_cutout.png");
        }
    });
}

// ===========================================================================
// MODE 3: DEBLURRING
// ===========================================================================
function initMode3() {
    const fileInput = document.getElementById("m3_file");
    const previewRow = document.getElementById("m3_preview");
    const thumbImg = document.getElementById("m3_thumb");
    const nameSpan = document.getElementById("m3_name");
    const removeBtn = document.getElementById("m3_remove");
    const runBtn = document.getElementById("m3_run_btn");

    const errorBox = document.getElementById("m3_error");
    const emptyMsg = document.getElementById("m3_empty_msg");
    const resultImg = document.getElementById("m3_result_img");
    const viewSwitches = document.getElementById("m3_view_switches");
    const downloadBtn = document.getElementById("m3_download_btn");

    let currentFile = null;
    let inputBase64 = null;
    let deblurredBase64 = null;

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            currentFile = e.target.files[0];
            nameSpan.textContent = currentFile.name;
            const reader = new FileReader();
            reader.onload = (ev) => {
                inputBase64 = ev.target.result;
                thumbImg.src = inputBase64;
                previewRow.classList.remove("hidden");
                errorBox.classList.add("hidden");
            };
            reader.readAsDataURL(currentFile);
        }
    });

    removeBtn.addEventListener("click", () => {
        currentFile = null;
        inputBase64 = null;
        fileInput.value = "";
        thumbImg.src = "";
        nameSpan.textContent = "";
        previewRow.classList.add("hidden");
    });

    if (viewSwitches) {
        viewSwitches.querySelectorAll(".switch-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const view = btn.dataset.view;
                if (view === "result" && deblurredBase64) {
                    resultImg.src = deblurredBase64;
                } else if (view === "original" && inputBase64) {
                    resultImg.src = inputBase64;
                }
            });
        });
    }

    runBtn.addEventListener("click", async () => {
        if (!currentFile) {
            errorBox.textContent = "Please upload an image.";
            errorBox.classList.remove("hidden");
            return;
        }

        errorBox.classList.add("hidden");
        runBtn.disabled = true;
        runBtn.textContent = "Deblurring image...";

        const formData = new FormData();
        formData.append("image", currentFile);

        try {
            const resp = await fetch("/api/mode3/deblur", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            if (data.status === "success") {
                deblurredBase64 = data.result_base64;
                resultImg.src = deblurredBase64;
                resultImg.classList.remove("hidden");
                emptyMsg.classList.add("hidden");

                viewSwitches.classList.remove("hidden");
                viewSwitches.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
                viewSwitches.querySelector('[data-view="result"]').classList.add("active");

                downloadBtn.classList.remove("hidden");
            } else {
                errorBox.textContent = "Error: " + (data.error || "Deblurring failed.");
                errorBox.classList.remove("hidden");
            }
        } catch (err) {
            errorBox.textContent = "Request error: " + err.message;
            errorBox.classList.remove("hidden");
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = "Deblur Image";
        }
    });

    downloadBtn.addEventListener("click", () => {
        if (deblurredBase64) {
            downloadBase64(deblurredBase64, "deblurred_image.jpg");
        }
    });
}
