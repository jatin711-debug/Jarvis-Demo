param([string]$OutputPath, [string]$WindowTitle)
Add-Type -AssemblyName System.Windows.Forms
$form = New-Object System.Windows.Forms.Form
$form.Text = $WindowTitle
$form.Width = 450
$form.Height = 180
$editor = New-Object System.Windows.Forms.TextBox
$editor.AccessibleName = 'Smoke text'
$editor.SetBounds(20, 20, 390, 25)
$button = New-Object System.Windows.Forms.Button
$button.Text = 'Save smoke file'
$button.SetBounds(20, 60, 160, 30)
$label = New-Object System.Windows.Forms.Label
$label.SetBounds(200, 65, 200, 30)
$button.Add_Click({
    [System.IO.File]::WriteAllText($OutputPath, $editor.Text)
    $label.Text = 'Saved successfully'
})
$form.Controls.AddRange(@($editor, $button, $label))
[System.Windows.Forms.Application]::Run($form)
