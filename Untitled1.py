#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import glob
import io
import py3Dmol
import ipywidgets as widgets
from IPython.display import display
from Bio.PDB import PDBParser, PDBIO, Superimposer

# 1. Find all matching PDB files in the correct directory
files = sorted(glob.glob("data/pdbs_by_group/others/TIM-DBD_x_UHRF1-TTD/*.pdb"))

if not files:
    print("No matching PDB files found.")
else:
    print(f"Processing and aligning {len(files)} files on Chain A...")

    parser = PDBParser(QUIET=True)
    structures = []

    # 2. Load all structures into Biopython
    for f in files:
        struct = parser.get_structure(f, f)
        structures.append(struct)

    # 3. Use the first structure as the alignment reference (Chain A only)
    ref_struct = structures[0]

    # Build a dictionary mapping residue ID -> CA Atom for Chain A of the reference
    ref_ca_dict = {}
    for chain in ref_struct.get_chains():
        if chain.id.upper() == 'A':
            for res in chain.get_residues():
                if "CA" in res:
                    ref_ca_dict[res.id[1]] = res["CA"]

    # 4. Align all subsequent models to the reference using matched Alpha-Carbons (CA) on Chain A only
    for i, struct in enumerate(structures[1:], start=1):
        mob_ca_dict = {}
        for chain in struct.get_chains():
            if chain.id.upper() == 'A':
                for res in chain.get_residues():
                    if "CA" in res:
                        mob_ca_dict[res.id[1]] = res["CA"]

        # Find common residues present on Chain A in both reference and mobile structure
        common_keys = sorted(list(set(ref_ca_dict.keys()).intersection(set(mob_ca_dict.keys()))))

        if not common_keys:
            print(f"Warning: No common CA atoms found on Chain A between reference and {files[i]}. Alignment skipped.")
            continue

        ref_atoms_matched = [ref_ca_dict[k] for k in common_keys]
        mob_atoms_matched = [mob_ca_dict[k] for k in common_keys]

        print(f"Aligning Model {i} ({files[i].split('/')[-1]}) using {len(common_keys)} matched CA atoms on Chain A.")

        sup = Superimposer()
        sup.set_atoms(ref_atoms_matched, mob_atoms_matched)
        sup.apply(struct.get_atoms())

    # 5. Initialize py3Dmol viewer and add aligned models
    view = py3Dmol.view(width=800, height=600)
    io_handler = PDBIO()

    model_styles = []

    for i, (f, struct) in enumerate(zip(files, structures)):
        io_handler.set_structure(struct)
        handle = io.StringIO()
        io_handler.save(handle)

        # Add model to py3Dmol viewer
        view.addModel(handle.getvalue(), "pdb")

        # Determine explicit colors based on filename keywords
        # 'abin': A: salmon, B: teal
        # 'model': A: grey, B: black
        # 'pdbeval': A: red, B: blue
        f_lower = f.lower()
        if "abin" in f_lower:
            color_A = "salmon"
            color_B = "teal"
        elif "model" in f_lower:
            color_A = "grey"
            color_B = "black"
        elif "pdbeval" in f_lower:
            color_A = "red"
            color_B = "blue"
        else:
            color_A = "orange"
            color_B = "purple"

        model_styles.append((color_A, color_B))

        # Apply chain-specific styling for both chains of this model
        view.setStyle({"model": i, "chain": "A"}, {"cartoon": {"color": color_A}})
        view.setStyle({"model": i, "chain": "B"}, {"cartoon": {"color": color_B}})

    view.zoomTo({})

    # 6. Interactive Controls (Model Checkboxes)
    checkboxes = []

    def make_model_toggle_handler(model_index, color_a, color_b):
        def on_change(change):
            if change['new']:
                view.setStyle({"model": model_index, "chain": "A"}, {"cartoon": {"color": color_a}})
                view.setStyle({"model": model_index, "chain": "B"}, {"cartoon": {"color": color_b}})
            else:
                view.setStyle({"model": model_index, "chain": "A"}, {"cartoon": {"hidden": True}})
                view.setStyle({"model": model_index, "chain": "B"}, {"cartoon": {"hidden": True}})
            view.update()
        return on_change

    # Build individual checkboxes per model/complex
    for i, (f, (cA, cB)) in enumerate(zip(files, model_styles)):
        cb = widgets.Checkbox(value=True, description=f"Model {i}: {f.split('/')[-1]}", layout=widgets.Layout(width='auto'))
        cb.observe(make_model_toggle_handler(i, cA, cB), names='value')
        checkboxes.append(cb)

    # Master "Select All / Deselect All" checkbox
    select_all_cb = widgets.Checkbox(value=True, description="<b>Select All / Deselect All</b>", layout=widgets.Layout(width='auto'))

    def on_select_all_change(change):
        for cb in checkboxes:
            cb.value = change['new']

    select_all_cb.observe(on_select_all_change, names='value')

    # 7. Enable Click-to-Display Model Name & Details on the 3D Viewer
    files_json = json.dumps([f.split('/')[-1] for f in files])

    click_js_callback = f"""
    function(atom, viewer, event, container) {{
        viewer.removeAllLabels();
        var fileList = {files_json};
        var fileName = fileList[atom.model] !== undefined ? fileList[atom.model] : "Unknown";
        var info = "Model " + atom.model + " (" + fileName + ") | Chain " + atom.chain + " | Res: " + atom.resn + " " + atom.resi;
        viewer.addLabel(info, {{
            position: atom,
            backgroundColor: "black",
            fontColor: "white",
            fontSize: 12
        }});
        viewer.render();
    }}
    """
    view.setClickable({}, True, click_js_callback)

    # 8. Separate Action Buttons Configuration with Fixed Zoom Call
    reset_view_btn = widgets.Button(
        description="🔍 Reset View & Labels",
        button_style='info',
        layout=widgets.Layout(width='48%')
    )

    def on_reset_view_clicked(b):
        view.removeAllLabels()
        view.zoomTo({})
        view.update()
        print("Camera zoom reset and labels cleared.")

    reset_view_btn.on_click(on_reset_view_clicked)

    show_all_btn = widgets.Button(
        description="👁️ Show All Models",
        button_style='success',
        layout=widgets.Layout(width='48%')
    )

    def on_show_all_clicked(b):
        select_all_cb.value = True
        for i, (cb, (cA, cB)) in enumerate(zip(checkboxes, model_styles)):
            cb.value = True
            view.setStyle({"model": i, "chain": "A"}, {"cartoon": {"color": cA}})
            view.setStyle({"model": i, "chain": "B"}, {"cartoon": {"color": cB}})
        view.zoomTo({})
        view.update()
        print("All models shown, colors restored, and view reset.")

    show_all_btn.on_click(on_show_all_clicked)

    # Group the two buttons side by side
    button_box = widgets.HBox([reset_view_btn, show_all_btn], layout=widgets.Layout(width='100%', justify_content='space-between', margin='0px 0px 5px 0px'))

    # Display controls panel and 3D viewer
    controls = widgets.VBox([button_box, select_all_cb] + checkboxes)

    display(controls)
    view.show()


# In[ ]:





# In[ ]:




